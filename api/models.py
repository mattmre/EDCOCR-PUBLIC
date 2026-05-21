"""Pydantic request/response schemas for the OCR API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class JobSubmitRequest(BaseModel):
    """Body fields for job submission (used alongside optional file upload)."""

    source_path: Optional[str] = Field(
        None, description="Server-side path to source document."
    )
    priority: Literal["urgent", "normal", "low"] = "normal"
    enable_docintel: bool = False
    docintel_mode: Literal["layout_only", "tables_only", "full"] = "full"
    skip_ocr: bool = Field(
        False, description="Skip the primary OCR engine step. Only supported if the document is a native PDF with embedded text or if relying entirely on NLP extraction."
    )
    processing_timeout_minutes: Optional[int] = Field(
        None,
        ge=1,
        description="Optional per-job processing timeout override in minutes.",
    )
    webhook_url: Optional[str] = Field(
        None, description="HTTPS URL to receive job completion notifications."
    )
    webhook_secret: Optional[str] = Field(
        None,
        description="HMAC secret for signing webhook payloads (write-only).",
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


class JobLinks(BaseModel):
    self_url: str = Field(..., alias="self")
    result: str

    model_config = {"populate_by_name": True}


class JobProgress(BaseModel):
    total_pages: int = 0
    pages_completed: int = 0
    percent_complete: float = 0.0
    current_stage: str = "submitted"


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    priority: str
    source_file: str
    estimated_pages: Optional[int] = None
    links: JobLinks


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: str
    source_file: str
    progress: Optional[JobProgress] = None
    settings: dict[str, Any] = {}
    webhook_status: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
    limit: int
    offset: int
    # Backward-compatible aliases (deprecated -- prefer limit/offset)
    page: Optional[int] = None
    per_page: Optional[int] = None


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    artifacts: dict[str, str] = {}
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    jobs: dict[str, int] = {}


class SubsystemCheck(BaseModel):
    """Individual subsystem health check result."""

    status: str  # "healthy", "degraded", "unhealthy"
    message: str = ""
    latency_ms: Optional[float] = None


class DetailedHealthResponse(HealthResponse):
    """Extended health response with subsystem checks."""

    checks: dict[str, SubsystemCheck] = {}


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Output manifest
# ---------------------------------------------------------------------------


class OutputArtifactResponse(BaseModel):
    """Single output artifact metadata."""

    output_type: str
    filename: str
    relative_path: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    schema_version: str = ""


class OutputManifestResponse(BaseModel):
    """Manifest of all outputs produced by a job."""

    job_id: str
    artifacts: list[OutputArtifactResponse]
    schema_versions: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Schema listing
# ---------------------------------------------------------------------------


class SchemaListItem(BaseModel):
    """Summary of a single schema definition."""

    output_type: str
    schema_version: str


class SchemaListResponse(BaseModel):
    """List of all available output schemas."""

    schemas: list[SchemaListItem]


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


class ReviewItemResponse(BaseModel):
    """Single review queue item."""

    review_id: str
    job_id: str
    reason: str
    confidence: float
    quality_classification: str
    status: str
    reviewer: str
    decision_notes: str
    created_at: str
    reviewed_at: str
    metadata: dict[str, Any] = {}


class ReviewQueueResponse(BaseModel):
    """Paginated list of review queue items."""

    items: list[ReviewItemResponse]
    total: int


class ReviewStatsResponse(BaseModel):
    """Review queue aggregate statistics."""

    pending: int
    approved: int
    rejected: int
    reprocess: int
    total: int
    avg_review_seconds: float = 0.0
    oldest_pending: str = ""


class ReviewDecisionRequest(BaseModel):
    """Body for submitting a review decision."""

    status: Literal["approved", "rejected", "reprocess"]
    reviewer: str = ""
    notes: str = ""


class ReviewCertifyRequest(BaseModel):
    """Body for certifying an approved review item with strong-auth evidence."""

    auth_method: Literal["piv_cac", "oidc_mfa", "hardware_token"]
    auth_token: str = Field(..., min_length=1)
    notes: str = ""


# ---------------------------------------------------------------------------
# Entity / extraction recall
# ---------------------------------------------------------------------------


class EntitySearchResult(BaseModel):
    """Single entity search result."""

    entity_id: str
    job_id: str
    entity_type: str
    text: str
    confidence: float
    source: str
    page: int
    document_name: str


class EntitySearchResponse(BaseModel):
    """Paginated entity search results."""

    results: list[EntitySearchResult]
    total: int
    limit: int
    offset: int


class ExtractionSearchResult(BaseModel):
    """Single extraction search result."""

    extraction_id: str
    job_id: str
    field_name: str
    field_value: str
    confidence: float
    page: int
    document_name: str


class ExtractionSearchResponse(BaseModel):
    """Paginated extraction search results."""

    results: list[ExtractionSearchResult]
    total: int
    limit: int
    offset: int


class RecallStatsResponse(BaseModel):
    """Entity/extraction index statistics."""

    total_entities: int
    total_extractions: int
    unique_entity_types: int
    unique_field_names: int
    jobs_indexed: int
