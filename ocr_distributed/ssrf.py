"""SSRF protection utilities for webhook URL validation.

Shared module used by both the REST API (api/webhooks.py) and the
distributed coordinator (coordinator/jobs/tasks.py).
"""

import logging
import socket
import urllib.error
import urllib.request
from ipaddress import ip_address
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP handler that blocks redirects to prevent SSRF bypass."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, f"Redirect blocked (SSRF protection): {newurl}",
            headers, fp,
        )


# Module-level opener that blocks redirects (SSRF protection)
safe_opener = urllib.request.build_opener(NoRedirectHandler)


def is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/loopback IP address."""
    try:
        addr = ip_address(hostname)
        return addr.is_private or addr.is_loopback or addr.is_reserved
    except ValueError:
        pass
    # Resolve hostname to check IP
    try:
        resolved = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for family, _type, _proto, _canonname, sockaddr in resolved:
            addr = ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_reserved:
                return True
    except (socket.gaierror, OSError):
        # Cannot resolve -- treat as potentially private for safety
        return True
    return False


def validate_webhook_url(
    url: str,
    *,
    allow_http: bool = False,
    allow_private: bool = False,
) -> str:
    """Validate a webhook URL for SSRF safety.

    Raises ValueError if URL is invalid or targets private infrastructure.

    - Must be HTTPS (unless allow_http=True for development)
    - Max 2048 characters
    - No private/loopback IPs (unless allow_private=True)
    """
    if not url or not url.strip():
        raise ValueError("Webhook URL must not be empty.")

    url = url.strip()
    if len(url) > 2048:
        raise ValueError("Webhook URL exceeds maximum length of 2048 characters.")

    parsed = urlparse(url)

    if not parsed.scheme:
        raise ValueError("Webhook URL must include a scheme (https://).")

    if parsed.scheme == "http" and not allow_http:
        raise ValueError(
            "Webhook URL must use HTTPS. Set WEBHOOK_ALLOW_HTTP=true for development."
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL has unsupported scheme: {parsed.scheme}")

    if not parsed.hostname:
        raise ValueError("Webhook URL must include a hostname.")

    hostname = parsed.hostname.lower()
    if hostname in LOOPBACK_NAMES:
        if not allow_private:
            raise ValueError(
                "Webhook URL must not point to localhost or loopback addresses."
            )

    if not allow_private and is_private_ip(hostname):
        raise ValueError(
            "Webhook URL must not point to private or reserved IP addresses."
        )

    return url
