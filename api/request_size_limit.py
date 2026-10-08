"""Request body size limit middleware (maintenance review).

Rejects oversized non-multipart requests before application handlers read them.
Both declared ``Content-Length`` and chunked ASGI request bodies are bounded.
Multipart uploads retain their separate ``MAX_UPLOAD_SIZE_MB`` policy.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class _RequestBodyTooLarge(Exception):
    """Internal control-flow signal for chunked request rejection."""


class RequestSizeLimitMiddleware:
    """Reject oversized requests using declared and streamed byte counts."""

    def __init__(self, app, max_size: int = 10 * 1024 * 1024) -> None:
        self.app = app
        self.max_size = max_size

    async def _send_error(self, send, status_code: int) -> None:
        content = {
            "error": "request_too_large"
            if status_code == 413
            else "invalid_content_length",
            "message": (
                f"Request body exceeds maximum allowed size of {self.max_size} bytes."
                if status_code == 413
                else "Malformed Content-Length header."
            ),
        }
        if status_code == 413:
            content["max_size"] = self.max_size
        body = json.dumps(content).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or self.max_size <= 0:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        # Multipart uploads have a separate application-level upload policy.
        if headers.get(b"content-type", b"").lower().startswith(b"multipart/"):
            await self.app(scope, receive, send)
            return

        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                content_length = int(content_length_raw)
            except (TypeError, ValueError):
                await self._send_error(send, 400)
                return
            if content_length < 0:
                await self._send_error(send, 400)
                return
            if content_length > self.max_size:
                logger.warning(
                    "Rejected oversized request: %d bytes > %d bytes",
                    content_length,
                    self.max_size,
                )
                await self._send_error(send, 413)
                return

        total_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal total_bytes
            message = await receive()
            if message.get("type") == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > self.max_size:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if not response_started:
                await self._send_error(send, 413)
            else:
                raise
