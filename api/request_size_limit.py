"""Request body size limit middleware.

Rejects requests whose declared ``Content-Length`` exceeds a configured
maximum before any downstream handler reads the body.  This blocks cheap
DoS vectors against JSON endpoints that do not perform their own
streaming/upload-aware size checks.

File-upload endpoints (``multipart/form-data``) are exempt because they
are already gated by ``MAX_UPLOAD_SIZE_MB`` at the application layer.

Configuration
-------------
``MAX_REQUEST_BODY_SIZE`` env var (bytes).  Default: ``10485760`` (10 MB).
Set to ``0`` to disable the check (not recommended).
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests based on the ``Content-Length`` header.

    The middleware is intentionally conservative:

    - Requests without a ``Content-Length`` header are passed through
      (chunked/streaming bodies are still gated by the underlying
      ASGI server and endpoint-level validation).
    - Multipart uploads are exempt so the existing upload size policy
      (``MAX_UPLOAD_SIZE_MB``) remains authoritative for file ingest.
    - An oversized request returns HTTP ``413 Payload Too Large`` with
      a structured JSON body so API clients get a machine-readable error.
    """

    def __init__(self, app, max_size: int = 10 * 1024 * 1024) -> None:
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self.max_size <= 0:
            return await call_next(request)

        # Exempt multipart uploads -- they have their own size policy.
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("multipart/"):
            return await call_next(request)

        content_length_raw = request.headers.get("content-length")
        if content_length_raw is None:
            return await call_next(request)

        try:
            content_length = int(content_length_raw)
        except ValueError:
            # Malformed header -- reject rather than guess.
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_content_length",
                    "message": "Malformed Content-Length header.",
                },
            )

        if content_length > self.max_size:
            logger.warning(
                "Rejected oversized request: %d bytes > %d bytes (%s %s)",
                content_length,
                self.max_size,
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": (
                        f"Request body exceeds maximum allowed size "
                        f"of {self.max_size} bytes."
                    ),
                    "max_size": self.max_size,
                },
            )

        return await call_next(request)
