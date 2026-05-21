"""Python SDK client for EDCOCR API.

.. deprecated:: 1.2.1
    **This module is the legacy in-tree SDK and is retained only for
    backward compatibility with older internal scripts.** It exposes
    ``OcrClient`` / ``JobInfo`` / ``JobStatus`` enums that are NOT compatible
    with the canonical published SDK under ``sdk/python/src/edcocr_sdk``.

    **New code must use the published SDK instead:**

    .. code-block:: python

        # Install: pip install edcocr-sdk
        from edcocr_sdk import EDCOCRClient

        with EDCOCRClient("http://localhost:8000", api_key="my-key") as client:
            job = client.submit_job("document.pdf")
            result = client.wait_for_completion(job.job_id)

    Differences from the canonical SDK:
    - Uses ``OcrClient`` (legacy) vs ``EDCOCRClient`` (canonical).
    - ``JobStatus`` enum values differ from the canonical SDK.
    - Does not expose ``Priority``, ``DocIntelMode``, ``BatchSubmission``,
      ``JobProgress``, or the async ``AsyncEDCOCRClient``.
    - No ``py.typed`` marker; type checkers see it as untyped.

    This file will be removed in a future major release. Do not add new
    features here; migrate callers to ``sdk/python/src/edcocr_sdk``.

Usage (legacy, for backward compat only):
    from sdk.python_client import OcrClient

    client = OcrClient("http://localhost:8000", api_key="my-key")
    job = client.submit("document.pdf")
    result = client.wait_for_result(job.job_id)
    client.download_result(job.job_id, "output.pdf")
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Union

from ocr_local.config.version import __version__ as _sdk_version

logger = logging.getLogger(__name__)

warnings.warn(
    "sdk.python_client is the legacy in-tree SDK and is deprecated. "
    "Use the canonical SDK: `from edcocr_sdk import EDCOCRClient` "
    "(see sdk/python/src/edcocr_sdk). This module will be removed in a "
    "future major release.",
    DeprecationWarning,
    stacklevel=2,
)

# Use lazy import for requests to allow tests without it
_requests = None


def _get_requests():
    global _requests
    if _requests is None:
        try:
            import requests as req
            _requests = req
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for the Python SDK. "
                "Install it with: pip install requests"
            )
    return _requests


class JobStatus(Enum):
    """Possible states of an OCR job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """OCR job information."""

    job_id: str = ""
    status: str = ""
    filename: str = ""
    pages: int = 0
    created_at: str = ""
    completed_at: str = ""
    error: str = ""
    progress: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "JobInfo":
        """Create a JobInfo from an API response dict.

        Handles both canonical and alternate key names so the SDK
        works even if the server uses slightly different field names.
        """
        return cls(
            job_id=data.get("job_id", data.get("id", "")),
            status=data.get("status", ""),
            filename=data.get("filename", data.get("original_filename", "")),
            pages=data.get("pages", data.get("total_pages", 0)),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error", data.get("error_message", "")),
            progress=data.get("progress", 0.0),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "filename": self.filename,
            "pages": self.pages,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "progress": self.progress,
        }

    @property
    def is_complete(self) -> bool:
        """Return True if the job has reached a terminal state."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def is_success(self) -> bool:
        """Return True if the job completed successfully."""
        return self.status == "completed"


@dataclass
class HealthInfo:
    """API health status."""

    status: str = ""
    version: str = ""
    uptime_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "HealthInfo":
        """Create from an API health response dict."""
        return cls(
            status=data.get("status", ""),
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", data.get("uptime", 0.0)),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "status": self.status,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
        }


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class OcrClientError(Exception):
    """Base exception for SDK errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(OcrClientError):
    """API key is invalid or missing."""

    pass


class NotFoundError(OcrClientError):
    """Resource not found."""

    pass


class TimeoutError(OcrClientError):
    """Operation timed out."""

    pass


class ServerError(OcrClientError):
    """Server returned a 5xx error."""

    pass


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------


class OcrClient:
    """Python SDK client for EDCOCR API.

    Args:
        base_url: Root URL of the OCR API (e.g. ``http://localhost:8000``).
        api_key: Value for the ``X-API-Key`` header.  Empty string skips auth.
        timeout: Default HTTP timeout in seconds.
        max_retries: Number of attempts on transient connection errors.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None

    @property
    def session(self):
        """Lazy-init requests session."""
        if self._session is None:
            requests = _get_requests()
            self._session = requests.Session()
            if self.api_key:
                self._session.headers["X-API-Key"] = self.api_key
            self._session.headers["User-Agent"] = f"ocr-local-python-sdk/{_sdk_version}"
        return self._session

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def health(self) -> HealthInfo:
        """Check API health."""
        resp = self._request("GET", "/api/v1/health")
        return HealthInfo.from_dict(resp)

    def submit(
        self,
        file_path: Union[str, Path] = None,
        file_obj: BinaryIO = None,
        filename: str = None,
        enable_docintel: bool = False,
        webhook_url: str = None,
        priority: str = None,
    ) -> JobInfo:
        """Submit a document for OCR processing.

        Args:
            file_path: Path to file to upload.
            file_obj: File-like object to upload (alternative to *file_path*).
            filename: Override filename (used with *file_obj*).
            enable_docintel: Enable document intelligence analysis.
            webhook_url: URL for completion webhook.
            priority: Job priority (``low``, ``normal``, ``high``).

        Returns:
            JobInfo with ``job_id`` for tracking.
        """
        if file_path is None and file_obj is None:
            raise ValueError("Either file_path or file_obj must be provided")

        data = {}
        if enable_docintel:
            data["enable_docintel"] = "true"
        if webhook_url:
            data["webhook_url"] = webhook_url
        if priority:
            data["priority"] = priority

        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            fname = filename or path.name
            with open(path, "rb") as f:
                files = {"file": (fname, f)}
                resp = self._request("POST", "/api/v1/jobs/", data=data, files=files)
        else:
            fname = filename or "upload"
            files = {"file": (fname, file_obj)}
            resp = self._request("POST", "/api/v1/jobs/", data=data, files=files)

        return JobInfo.from_dict(resp)

    def get_job(self, job_id: str) -> JobInfo:
        """Get current status of a job."""
        resp = self._request("GET", f"/api/v1/jobs/{job_id}")
        return JobInfo.from_dict(resp)

    def list_jobs(
        self,
        status: str = None,
        limit: int = None,
        offset: int = None,
    ) -> list:
        """List jobs with optional filtering.

        Returns:
            A list of :class:`JobInfo` objects.
        """
        params = {}
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        resp = self._request("GET", "/api/v1/jobs/", params=params)

        if isinstance(resp, list):
            return [JobInfo.from_dict(j) for j in resp]
        jobs = resp.get("jobs", resp.get("items", []))
        return [JobInfo.from_dict(j) for j in jobs]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job.

        Returns:
            ``True`` if the job was cancelled, ``False`` if it was not found.
        """
        try:
            self._request("DELETE", f"/api/v1/jobs/{job_id}")
            return True
        except NotFoundError:
            return False

    def download_result(
        self,
        job_id: str,
        output_path: Union[str, Path] = None,
    ) -> bytes:
        """Download the result of a completed job.

        Args:
            job_id: Job ID.
            output_path: Optional path to save the result file.

        Returns:
            Raw bytes of the result.
        """
        requests = _get_requests()
        url = f"{self.base_url}/api/v1/jobs/{job_id}/result"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
        self._check_response(resp)

        content = resp.content
        if output_path:
            Path(output_path).write_bytes(content)

        return content

    def get_outputs(self, job_id: str) -> dict:
        """Get output manifest for a completed job.

        Returns dict with 'job_id', 'artifacts' list, and 'schema_versions'.
        Each artifact has: output_type, filename, relative_path, size_bytes,
        schema_version.
        """
        resp = self._request("GET", f"/api/v1/jobs/{job_id}/outputs")
        return resp

    def get_output(self, job_id: str, output_type: str) -> bytes:
        """Download a specific output artifact.

        Args:
            job_id: Job identifier.
            output_type: One of: ocr_text, searchable_pdf, structure, entities,
                ner, extraction, classification, validation, handwriting,
                signature, vertical, custody.

        Returns:
            Raw bytes of the output file.
        """
        requests = _get_requests()
        url = f"{self.base_url}/api/v1/jobs/{job_id}/outputs/{output_type}"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        resp = requests.get(url, headers=headers, timeout=self.timeout)
        self._check_response(resp)
        return resp.content

    def get_output_json(self, job_id: str, output_type: str) -> dict:
        """Download a JSON output artifact and parse it.

        For JSON sidecar outputs (entities, ner, extraction, classification,
        validation, handwriting, signature, vertical, structure).
        """
        content = self.get_output(job_id, output_type)
        return json.loads(content)

    def list_schemas(self) -> list:
        """List available output schemas.

        Returns list of dicts with 'output_type' and 'schema_version'.
        """
        resp = self._request("GET", "/api/v1/schemas")
        return resp.get("schemas", [])

    def get_schema(self, output_type: str) -> dict:
        """Get JSON Schema definition for an output type."""
        resp = self._request("GET", f"/api/v1/schemas/{output_type}")
        return resp

    def wait_for_result(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> JobInfo:
        """Poll until job completes or times out.

        Args:
            job_id: Job ID to monitor.
            poll_interval: Seconds between polls.
            timeout: Maximum seconds to wait.

        Returns:
            Final :class:`JobInfo`.

        Raises:
            TimeoutError: If job doesn't complete within *timeout*.
        """
        start = time.time()
        while True:
            job = self.get_job(job_id)
            if job.is_complete:
                return job

            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(status: {job.status})"
                )

            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        file_path: Union[str, Path],
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        **submit_kwargs,
    ) -> JobInfo:
        """Submit a file and wait for completion.

        Convenience method combining :meth:`submit` and :meth:`wait_for_result`.
        """
        job = self.submit(file_path=file_path, **submit_kwargs)
        return self.wait_for_result(job.job_id, poll_interval, timeout)

    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with retries and error handling."""
        url = f"{self.base_url}{path}"

        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        last_error = None
        for attempt in range(max(1, self.max_retries)):
            try:
                resp = self.session.request(method, url, **kwargs)
                self._check_response(resp)

                if resp.status_code == 204:
                    return {}

                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type:
                    return resp.json()
                return {"raw": resp.text}

            except (ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
                continue

        raise OcrClientError(
            f"Request failed after {self.max_retries} retries: {last_error}"
        )

    def _check_response(self, resp):
        """Check HTTP response and raise appropriate errors."""
        if resp.status_code < 400:
            return

        body = ""
        try:
            body = resp.text
        except Exception:
            pass

        if resp.status_code in (401, 403):
            raise AuthenticationError(
                f"Authentication failed (HTTP {resp.status_code})",
                status_code=resp.status_code,
                response_body=body,
            )
        elif resp.status_code == 404:
            raise NotFoundError(
                "Resource not found (HTTP 404)",
                status_code=404,
                response_body=body,
            )
        elif resp.status_code >= 500:
            raise ServerError(
                f"Server error (HTTP {resp.status_code})",
                status_code=resp.status_code,
                response_body=body,
            )
        else:
            raise OcrClientError(
                f"Request failed (HTTP {resp.status_code})",
                status_code=resp.status_code,
                response_body=body,
            )
