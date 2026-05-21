"""Pydantic v2 models for EDCOCR API requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Possible states of an OCR job."""

    QUEUED = "queued"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Job priority levels."""

    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class DocIntelMode(str, Enum):
    """Document Intelligence analysis modes."""

    LAYOUT_ONLY = "layout_only"
    TABLES_ONLY = "tables_only"
    FULL = "full"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JobProgress(BaseModel):
    """Progress information for a running job."""

    total_pages: int = 0
    pages_completed: int = 0
    percent_complete: float = 0.0
    current_stage: str = "submitted"


class JobLinks(BaseModel):
    """HATEOAS links for a job resource."""

    self_url: str = Field(default="", alias="self")
    result: str = ""

    model_config = {"populate_by_name": True}


class Job(BaseModel):
    """Full job representation returned by status and list endpoints."""

    job_id: str = ""
    status: str = ""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: str = "normal"
    source_file: str = ""
    progress: Optional[JobProgress] = None
    settings: Dict[str, Any] = {}
    webhook_status: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        """Return True if the job has reached a terminal state."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def is_success(self) -> bool:
        """Return True if the job completed successfully."""
        return self.status == "completed"


class JobSubmitResult(BaseModel):
    """Response from job submission."""

    job_id: str = ""
    status: str = ""
    created_at: Optional[datetime] = None
    priority: str = "normal"
    source_file: str = ""
    estimated_pages: Optional[int] = None
    links: Optional[JobLinks] = None


class JobResult(BaseModel):
    """Result metadata for a completed job."""

    job_id: str = ""
    status: str = ""
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    artifacts: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: List[Job] = []
    total: int = 0
    limit: int = 50
    offset: int = 0
    # Backward-compatible aliases (deprecated -- prefer limit/offset)
    page: Optional[int] = None
    per_page: Optional[int] = None


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = ""
    version: str = ""
    uptime_seconds: float = 0.0
    jobs: Dict[str, int] = {}


class ErrorDetail(BaseModel):
    """Structured error response from the API."""

    error: str = ""
    message: str = ""
    details: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Submission parameters
# ---------------------------------------------------------------------------


class BatchItem(BaseModel):
    """A single item in a batch submission."""

    source_path: Optional[str] = None
    priority: Priority = Priority.NORMAL
    enable_docintel: bool = False
    docintel_mode: DocIntelMode = DocIntelMode.FULL
    webhook_url: Optional[str] = None


class BatchSubmission(BaseModel):
    """Batch of job submissions."""

    items: List[BatchItem] = []
