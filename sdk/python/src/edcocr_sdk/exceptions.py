"""Custom exception hierarchy for the EDCOCR SDK."""

from __future__ import annotations


class OCRLocalError(Exception):
    """Base exception for all EDCOCR SDK errors.

    Attributes:
        status_code: HTTP status code that triggered the error (0 if not HTTP-related).
        response_body: Raw response body text, if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.args[0]!r}, status_code={self.status_code})"


class AuthenticationError(OCRLocalError):
    """API key is invalid, missing, or the caller lacks permission (HTTP 401/403)."""


class NotFoundError(OCRLocalError):
    """The requested resource does not exist (HTTP 404)."""


class RateLimitError(OCRLocalError):
    """The server returned HTTP 429 -- too many requests or queue full."""


class ValidationError(OCRLocalError):
    """The server rejected the request due to invalid input (HTTP 400/422)."""


class ConflictError(OCRLocalError):
    """The request conflicts with the current resource state (HTTP 409)."""


class ServerError(OCRLocalError):
    """The server returned a 5xx error."""


class TimeoutError(OCRLocalError):
    """An operation timed out (polling or HTTP request)."""
