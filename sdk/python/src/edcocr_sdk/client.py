"""Synchronous and asynchronous HTTP clients for the EDCOCR API.

Usage (sync)::

    from edcocr_sdk import EDCOCRClient

    client = EDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"])
    job = client.submit_job("document.pdf")
    result = client.wait_for_completion(job.job_id)
    client.close()

Usage (async)::

    from edcocr_sdk import AsyncEDCOCRClient

    async with AsyncEDCOCRClient("http://localhost:8000", api_key=os.environ["OCR_API_KEY"]) as client:
        job = await client.submit_job("document.pdf")
        result = await client.wait_for_completion(job.job_id)
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Optional, Union

import httpx

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
    HealthResponse,
    Job,
    JobListResponse,
    JobResult,
    JobSubmitResult,
)

logger = logging.getLogger(__name__)

SDK_VERSION = "4.1.0"
USER_AGENT = f"edcocr-sdk-python/{SDK_VERSION}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _raise_for_status(response: httpx.Response) -> None:
    """Inspect an HTTP response and raise the appropriate SDK exception."""
    if response.status_code < 400:
        return

    body = ""
    try:
        body = response.text
    except Exception:
        pass

    status = response.status_code
    if status in (401, 403):
        raise AuthenticationError(
            f"Authentication failed (HTTP {status})",
            status_code=status,
            response_body=body,
        )
    if status == 404:
        raise NotFoundError(
            "Resource not found (HTTP 404)",
            status_code=404,
            response_body=body,
        )
    if status == 409:
        raise ConflictError(
            "Conflict (HTTP 409)",
            status_code=409,
            response_body=body,
        )
    if status == 422:
        raise ValidationError(
            "Validation error (HTTP 422)",
            status_code=422,
            response_body=body,
        )
    if status == 429:
        raise RateLimitError(
            "Rate limit exceeded (HTTP 429)",
            status_code=429,
            response_body=body,
        )
    if status >= 500:
        raise ServerError(
            f"Server error (HTTP {status})",
            status_code=status,
            response_body=body,
        )
    raise OCRLocalError(
        f"Request failed (HTTP {status})",
        status_code=status,
        response_body=body,
    )


def _build_headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"User-Agent": USER_AGENT}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _build_submit_data(
    *,
    enable_docintel: bool = False,
    docintel_mode: Optional[str] = None,
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    priority: Optional[str] = None,
    processing_timeout_minutes: Optional[int] = None,
) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if enable_docintel:
        data["enable_docintel"] = "true"
    if docintel_mode:
        data["docintel_mode"] = docintel_mode
    if webhook_url:
        data["webhook_url"] = webhook_url
    if webhook_secret:
        data["webhook_secret"] = webhook_secret
    if priority:
        data["priority"] = priority
    if processing_timeout_minutes is not None:
        data["processing_timeout_minutes"] = str(processing_timeout_minutes)
    return data


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------


class EDCOCRClient:
    """Synchronous Python client for the EDCOCR REST API.

    Args:
        base_url: Root URL of the OCR API (e.g. ``http://localhost:8000``).
        api_key: Value for the ``X-API-Key`` header. Empty string skips auth.
        timeout: Default HTTP timeout in seconds.
        max_retries: Number of retry attempts on transient connection errors.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-init httpx client."""
        if self._client is None or self._client.is_closed:
            transport = httpx.HTTPTransport(retries=self.max_retries)
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=_build_headers(self.api_key),
                timeout=self.timeout,
                transport=transport,
            )
        return self._client

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> HealthResponse:
        """Check API health status."""
        resp = self.client.get("/api/v1/health")
        _raise_for_status(resp)
        return HealthResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def submit_job(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
        source_path: Optional[str] = None,
        enable_docintel: bool = False,
        docintel_mode: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        priority: Optional[str] = None,
        processing_timeout_minutes: Optional[int] = None,
    ) -> JobSubmitResult:
        """Submit a document for OCR processing.

        Provide exactly one of *file_path*, *file_obj*, or *source_path*.

        Args:
            file_path: Local path to a file to upload.
            file_obj: Open file-like object to upload.
            filename: Override the uploaded filename (used with *file_obj*).
            source_path: Server-side path to source document.
            enable_docintel: Enable document intelligence analysis.
            docintel_mode: DocIntel mode (``layout_only``, ``tables_only``, ``full``).
            webhook_url: HTTPS URL for job completion webhook.
            webhook_secret: HMAC secret for webhook payload signing.
            priority: Job priority (``urgent``, ``normal``, ``low``).
            processing_timeout_minutes: Per-job timeout override in minutes.

        Returns:
            A :class:`JobSubmitResult` with the new ``job_id``.
        """
        if file_path is None and file_obj is None and source_path is None:
            raise ValueError(
                "Provide at least one of file_path, file_obj, or source_path"
            )

        data = _build_submit_data(
            enable_docintel=enable_docintel,
            docintel_mode=docintel_mode,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            priority=priority,
            processing_timeout_minutes=processing_timeout_minutes,
        )

        if source_path is not None:
            data["source_path"] = source_path

        files: Optional[Dict[str, Any]] = None
        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            fname = filename or path.name
            fh = open(path, "rb")
            files = {"file": (fname, fh)}
        elif file_obj is not None:
            fname = filename or "upload"
            files = {"file": (fname, file_obj)}

        try:
            resp = self.client.post("/api/v1/jobs", data=data, files=files)
        finally:
            if file_path is not None and files:
                files["file"][1].close()

        _raise_for_status(resp)
        return JobSubmitResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job status
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Job:
        """Get the current status of a job.

        Args:
            job_id: The job identifier.

        Returns:
            A :class:`Job` with full status and progress.
        """
        resp = self.client.get(f"/api/v1/jobs/{job_id}")
        _raise_for_status(resp)
        return Job.model_validate(resp.json())

    def list_jobs(
        self,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> JobListResponse:
        """List jobs with optional filtering and pagination.

        Args:
            status: Filter by job status.
            page: Page number (1-based).
            per_page: Results per page (max 100).

        Returns:
            A :class:`JobListResponse` with paginated jobs.
        """
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if status:
            params["status"] = status

        resp = self.client.get("/api/v1/jobs", params=params)
        _raise_for_status(resp)
        return JobListResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job results
    # ------------------------------------------------------------------

    def get_result(self, job_id: str) -> JobResult:
        """Get result metadata for a completed job.

        Args:
            job_id: The job identifier.

        Returns:
            A :class:`JobResult` with artifact links and metadata.
        """
        resp = self.client.get(f"/api/v1/jobs/{job_id}/result")
        _raise_for_status(resp)
        return JobResult.model_validate(resp.json())

    def download_artifact(
        self,
        job_id: str,
        artifact_type: str = "pdf",
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """Download a result artifact.

        Args:
            job_id: The job identifier.
            artifact_type: Artifact type (``pdf``, ``text``, ``structure``).
            output_path: Optional local path to save the downloaded file.

        Returns:
            Raw bytes of the artifact.
        """
        resp = self.client.get(
            f"/api/v1/jobs/{job_id}/result/download",
            params={"type": artifact_type},
        )
        _raise_for_status(resp)

        content = resp.content
        if output_path:
            Path(output_path).write_bytes(content)
        return content

    def list_outputs(self, job_id: str) -> Dict[str, Any]:
        """List generated output artifacts for a completed job."""
        resp = self.client.get(f"/api/v1/jobs/{job_id}/outputs")
        _raise_for_status(resp)
        return resp.json()

    def get_document_bundle(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the EDC DocumentBundle v1 for a completed OCR job."""
        resp = self.client.get(f"/api/v1/jobs/{job_id}/document-bundle")
        _raise_for_status(resp)
        return resp.json()

    def export_document_bundle(
        self,
        job_id: str,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Retrieve and write the EDC DocumentBundle v1 for a job."""
        bundle = self.get_document_bundle(job_id)
        Path(output_path).write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle

    def get_evidence_bundle(self, job_id: str) -> Dict[str, Any]:
        """Retrieve OCR custody/evidence metadata for a completed job."""
        resp = self.client.get(f"/api/v1/jobs/{job_id}/evidence-bundle")
        _raise_for_status(resp)
        return resp.json()

    def export_evidence_bundle(
        self,
        job_id: str,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Retrieve and write OCR custody/evidence metadata for a job."""
        bundle = self.get_evidence_bundle(job_id)
        Path(output_path).write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle

    def verify_custody(self, job_id: str) -> bool:
        """Return True when the job evidence bundle reports valid custody."""
        evidence = self.get_evidence_bundle(job_id)
        custody = evidence.get("custody", {})
        return bool(custody.get("available") and custody.get("valid"))

    # ------------------------------------------------------------------
    # Batch jobs
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        file_paths: Optional[Iterable[Union[str, Path]]] = None,
        source_paths: Optional[Iterable[str]] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit multiple documents for OCR processing as a batch."""
        paths = [Path(p) for p in file_paths or []]
        sources = list(source_paths or [])
        if not paths and not sources:
            raise ValueError("Provide at least one file_path or source_path")

        data: Dict[str, str] = {
            "priority": priority,
            "enable_docintel": str(enable_docintel).lower(),
            "docintel_mode": docintel_mode,
            "skip_ocr": str(skip_ocr).lower(),
        }
        if sources:
            data["source_paths"] = json.dumps(sources)
        if processing_timeout_minutes is not None:
            data["processing_timeout_minutes"] = str(processing_timeout_minutes)
        if webhook_url:
            data["webhook_url"] = webhook_url
        if webhook_secret:
            data["webhook_secret"] = webhook_secret

        files = []
        handles = []
        try:
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                handle = open(path, "rb")
                handles.append(handle)
                files.append(("files", (path.name, handle)))
            resp = self.client.post("/api/v1/jobs/batch", data=data, files=files or None)
        finally:
            for handle in handles:
                handle.close()

        _raise_for_status(resp)
        return resp.json()

    def list_batches(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List OCR batches with optional status filtering."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        resp = self.client.get("/api/v1/jobs/batch", params=params)
        _raise_for_status(resp)
        return resp.json()

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get status and child job summaries for a batch."""
        resp = self.client.get(f"/api/v1/jobs/batch/{batch_id}")
        _raise_for_status(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Job actions
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str) -> Job:
        """Cancel a queued or processing job.

        Args:
            job_id: The job identifier.

        Returns:
            Updated :class:`Job` with cancelled status.
        """
        resp = self.client.delete(f"/api/v1/jobs/{job_id}")
        _raise_for_status(resp)
        return Job.model_validate(resp.json())

    def retry_job(self, job_id: str) -> JobSubmitResult:
        """Retry a failed or cancelled job.

        Args:
            job_id: The original job identifier.

        Returns:
            A :class:`JobSubmitResult` for the new retry job.
        """
        resp = self.client.post(f"/api/v1/jobs/{job_id}/retry")
        _raise_for_status(resp)
        return JobSubmitResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> Job:
        """Poll until a job reaches a terminal state.

        Args:
            job_id: Job to monitor.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising :class:`TimeoutError`.

        Returns:
            The final :class:`Job` state.

        Raises:
            TimeoutError: If the job does not complete within *timeout* seconds.
        """
        start = time.monotonic()
        while True:
            job = self.get_job(job_id)
            if job.is_terminal:
                return job

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(last status: {job.status})"
                )
            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        file_path: Union[str, Path],
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        **submit_kwargs: Any,
    ) -> Job:
        """Submit a file and wait for completion.

        Convenience wrapper combining :meth:`submit_job` and
        :meth:`wait_for_completion`.
        """
        result = self.submit_job(file_path=file_path, **submit_kwargs)
        return self.wait_for_completion(result.job_id, poll_interval, timeout)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __enter__(self) -> "EDCOCRClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Asynchronous client
# ---------------------------------------------------------------------------


class AsyncEDCOCRClient:
    """Asynchronous Python client for the EDCOCR REST API.

    Uses ``httpx.AsyncClient`` under the hood for non-blocking I/O.

    Args:
        base_url: Root URL of the OCR API (e.g. ``http://localhost:8000``).
        api_key: Value for the ``X-API-Key`` header. Empty string skips auth.
        timeout: Default HTTP timeout in seconds.
        max_retries: Number of retry attempts on transient connection errors.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-init httpx async client."""
        if self._client is None or self._client.is_closed:
            transport = httpx.AsyncHTTPTransport(retries=self.max_retries)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=_build_headers(self.api_key),
                timeout=self.timeout,
                transport=transport,
            )
        return self._client

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> HealthResponse:
        """Check API health status."""
        resp = await self.client.get("/api/v1/health")
        _raise_for_status(resp)
        return HealthResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        file_path: Optional[Union[str, Path]] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
        source_path: Optional[str] = None,
        enable_docintel: bool = False,
        docintel_mode: Optional[str] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        priority: Optional[str] = None,
        processing_timeout_minutes: Optional[int] = None,
    ) -> JobSubmitResult:
        """Submit a document for OCR processing.

        See :meth:`EDCOCRClient.submit_job` for parameter details.
        """
        if file_path is None and file_obj is None and source_path is None:
            raise ValueError(
                "Provide at least one of file_path, file_obj, or source_path"
            )

        data = _build_submit_data(
            enable_docintel=enable_docintel,
            docintel_mode=docintel_mode,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            priority=priority,
            processing_timeout_minutes=processing_timeout_minutes,
        )

        if source_path is not None:
            data["source_path"] = source_path

        files: Optional[Dict[str, Any]] = None
        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            fname = filename or path.name
            fh = open(path, "rb")
            files = {"file": (fname, fh)}
        elif file_obj is not None:
            fname = filename or "upload"
            files = {"file": (fname, file_obj)}

        try:
            resp = await self.client.post("/api/v1/jobs", data=data, files=files)
        finally:
            if file_path is not None and files:
                files["file"][1].close()

        _raise_for_status(resp)
        return JobSubmitResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job status
    # ------------------------------------------------------------------

    async def get_job(self, job_id: str) -> Job:
        """Get the current status of a job."""
        resp = await self.client.get(f"/api/v1/jobs/{job_id}")
        _raise_for_status(resp)
        return Job.model_validate(resp.json())

    async def list_jobs(
        self,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> JobListResponse:
        """List jobs with optional filtering and pagination."""
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if status:
            params["status"] = status

        resp = await self.client.get("/api/v1/jobs", params=params)
        _raise_for_status(resp)
        return JobListResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Job results
    # ------------------------------------------------------------------

    async def get_result(self, job_id: str) -> JobResult:
        """Get result metadata for a completed job."""
        resp = await self.client.get(f"/api/v1/jobs/{job_id}/result")
        _raise_for_status(resp)
        return JobResult.model_validate(resp.json())

    async def download_artifact(
        self,
        job_id: str,
        artifact_type: str = "pdf",
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """Download a result artifact."""
        resp = await self.client.get(
            f"/api/v1/jobs/{job_id}/result/download",
            params={"type": artifact_type},
        )
        _raise_for_status(resp)

        content = resp.content
        if output_path:
            Path(output_path).write_bytes(content)
        return content

    async def list_outputs(self, job_id: str) -> Dict[str, Any]:
        """List generated output artifacts for a completed job."""
        resp = await self.client.get(f"/api/v1/jobs/{job_id}/outputs")
        _raise_for_status(resp)
        return resp.json()

    async def get_document_bundle(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the EDC DocumentBundle v1 for a completed OCR job."""
        resp = await self.client.get(f"/api/v1/jobs/{job_id}/document-bundle")
        _raise_for_status(resp)
        return resp.json()

    async def export_document_bundle(
        self,
        job_id: str,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Retrieve and write the EDC DocumentBundle v1 for a job."""
        bundle = await self.get_document_bundle(job_id)
        Path(output_path).write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle

    async def get_evidence_bundle(self, job_id: str) -> Dict[str, Any]:
        """Retrieve OCR custody/evidence metadata for a completed job."""
        resp = await self.client.get(f"/api/v1/jobs/{job_id}/evidence-bundle")
        _raise_for_status(resp)
        return resp.json()

    async def export_evidence_bundle(
        self,
        job_id: str,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Retrieve and write OCR custody/evidence metadata for a job."""
        bundle = await self.get_evidence_bundle(job_id)
        Path(output_path).write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle

    async def verify_custody(self, job_id: str) -> bool:
        """Return True when the job evidence bundle reports valid custody."""
        evidence = await self.get_evidence_bundle(job_id)
        custody = evidence.get("custody", {})
        return bool(custody.get("available") and custody.get("valid"))

    # ------------------------------------------------------------------
    # Batch jobs
    # ------------------------------------------------------------------

    async def submit_batch(
        self,
        file_paths: Optional[Iterable[Union[str, Path]]] = None,
        source_paths: Optional[Iterable[str]] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit multiple documents for OCR processing as a batch."""
        paths = [Path(p) for p in file_paths or []]
        sources = list(source_paths or [])
        if not paths and not sources:
            raise ValueError("Provide at least one file_path or source_path")

        data: Dict[str, str] = {
            "priority": priority,
            "enable_docintel": str(enable_docintel).lower(),
            "docintel_mode": docintel_mode,
            "skip_ocr": str(skip_ocr).lower(),
        }
        if sources:
            data["source_paths"] = json.dumps(sources)
        if processing_timeout_minutes is not None:
            data["processing_timeout_minutes"] = str(processing_timeout_minutes)
        if webhook_url:
            data["webhook_url"] = webhook_url
        if webhook_secret:
            data["webhook_secret"] = webhook_secret

        files = []
        handles = []
        try:
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                handle = open(path, "rb")
                handles.append(handle)
                files.append(("files", (path.name, handle)))
            resp = await self.client.post(
                "/api/v1/jobs/batch",
                data=data,
                files=files or None,
            )
        finally:
            for handle in handles:
                handle.close()

        _raise_for_status(resp)
        return resp.json()

    async def list_batches(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List OCR batches with optional status filtering."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        resp = await self.client.get("/api/v1/jobs/batch", params=params)
        _raise_for_status(resp)
        return resp.json()

    async def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get status and child job summaries for a batch."""
        resp = await self.client.get(f"/api/v1/jobs/batch/{batch_id}")
        _raise_for_status(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Job actions
    # ------------------------------------------------------------------

    async def cancel_job(self, job_id: str) -> Job:
        """Cancel a queued or processing job."""
        resp = await self.client.delete(f"/api/v1/jobs/{job_id}")
        _raise_for_status(resp)
        return Job.model_validate(resp.json())

    async def retry_job(self, job_id: str) -> JobSubmitResult:
        """Retry a failed or cancelled job."""
        resp = await self.client.post(f"/api/v1/jobs/{job_id}/retry")
        _raise_for_status(resp)
        return JobSubmitResult.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Polling helpers
    # ------------------------------------------------------------------

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> Job:
        """Poll until a job reaches a terminal state.

        Raises:
            TimeoutError: If the job does not complete within *timeout* seconds.
        """
        import asyncio

        start = time.monotonic()
        while True:
            job = await self.get_job(job_id)
            if job.is_terminal:
                return job

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(last status: {job.status})"
                )
            await asyncio.sleep(poll_interval)

    async def submit_and_wait(
        self,
        file_path: Union[str, Path],
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        **submit_kwargs: Any,
    ) -> Job:
        """Submit a file and wait for completion."""
        result = await self.submit_job(file_path=file_path, **submit_kwargs)
        return await self.wait_for_completion(
            result.job_id, poll_interval, timeout
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncEDCOCRClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
