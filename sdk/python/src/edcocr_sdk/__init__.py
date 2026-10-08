"""OCR_LOCAL SDK -- Python client for the forensic-grade OCR pipeline.

Quick start (set OCR_API_KEY in the environment)::

    from edcocr_sdk import EDCOCRClient

    with EDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
        job = client.submit_job("document.pdf")
        result = client.wait_for_completion(job.job_id)
        print(result.status)

For async usage::

    from edcocr_sdk import AsyncEDCOCRClient

    async with AsyncEDCOCRClient(
        "http://localhost:8000", api_key=os.environ["OCR_API_KEY"]
    ) as client:
        job = await client.submit_job("document.pdf")
        result = await client.wait_for_completion(job.job_id)
"""

from edcocr_sdk.client import AsyncEDCOCRClient, EDCOCRClient
from edcocr_sdk.exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    OCRLocalError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from edcocr_sdk.models import (
    BatchItem,
    BatchSubmission,
    DocIntelMode,
    ErrorDetail,
    HealthResponse,
    Job,
    JobListResponse,
    JobProgress,
    JobResult,
    JobStatus,
    JobSubmitResult,
    Priority,
)

__version__ = "4.2.0-rc.1"

__all__ = [
    # Clients
    "EDCOCRClient",
    "AsyncEDCOCRClient",
    # Models
    "BatchItem",
    "BatchSubmission",
    "DocIntelMode",
    "ErrorDetail",
    "HealthResponse",
    "Job",
    "JobListResponse",
    "JobProgress",
    "JobResult",
    "JobStatus",
    "JobSubmitResult",
    "Priority",
    # Exceptions
    "AuthenticationError",
    "ConflictError",
    "NotFoundError",
    "OCRLocalError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ValidationError",
]
