"""Pydantic request/response schemas for the batch job API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class BatchSubmitRequest(BaseModel):
    """Body fields for batch submission (used alongside file uploads)."""

    source_paths: Optional[str] = Field(
        None,
        description="JSON array of server-side paths to source documents.",
    )
    priority: Literal["urgent", "normal", "low"] = "normal"
    enable_docintel: bool = False
    docintel_mode: Literal["layout_only", "tables_only", "full"] = "full"
    processing_timeout_minutes: Optional[int] = Field(
        None,
        ge=1,
        description="Optional per-job processing timeout override in minutes.",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="HTTPS URL to receive batch completion notifications.",
    )
    webhook_secret: Optional[str] = Field(
        None,
        description="HMAC secret for signing webhook payloads (write-only).",
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


class BatchJobSummary(BaseModel):
    """Summary of a single job within a batch."""

    job_id: str
    source_file: str
    status: str


class BatchLinks(BaseModel):
    """HATEOAS links for batch responses."""

    self_url: str = Field(..., alias="self")
    jobs: str

    model_config = {"populate_by_name": True}


class BatchProgressInfo(BaseModel):
    """Aggregated progress information for a batch."""

    submitted: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    percent_complete: float = 0.0


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class BatchSubmitResponse(BaseModel):
    """Response returned when a batch is successfully submitted."""

    batch_id: str
    status: str
    created_at: datetime
    total_jobs: int
    priority: str
    jobs: list[BatchJobSummary]
    links: BatchLinks


class BatchStatusResponse(BaseModel):
    """Full batch status with progress and child job summaries."""

    batch_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    total_jobs: int
    progress: BatchProgressInfo
    jobs: list[BatchJobSummary]
    settings: dict[str, Any] = {}
    webhook_status: Optional[str] = None


class BatchListResponse(BaseModel):
    """Paginated list of batches."""

    batches: list[BatchStatusResponse]
    total: int
    limit: int
    offset: int
    # Backward-compatible aliases (deprecated -- prefer limit/offset)
    page: Optional[int] = None
    per_page: Optional[int] = None
