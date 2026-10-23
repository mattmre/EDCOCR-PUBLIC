"""Deterministic public DNS for webhook delivery tests.

networknone runners fail ``getaddrinfo`` before the mocked transport runs.
Production delivery still validates the URL and pins the resolved address, so
tests mock only that DNS boundary. RFC 5737 documentation ranges are reserved
(``ipaddress`` ``is_global`` is false) and must not be used as the pin.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from ipaddress import ip_address
from unittest.mock import patch

# Globally routable. Same address the SSRF pin tests already treat as public.
PUBLIC_WEBHOOK_PIN = "8.8.8.8"


def public_webhook_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Return one public A record, passing literal IPs through unchanged.

    Literal passthrough keeps private/reserved IP policy on the real
    ``ipaddress`` checks. Hostnames always answer with ``PUBLIC_WEBHOOK_PIN``
    so a missing resolver cannot fail the pin closed or sinkhole to a
    reserved address.
    """
    lookup = host
    if isinstance(host, bytes):
        lookup = host.decode("idna", errors="replace")
    lookup = str(lookup).strip().lower().rstrip(".")
    try:
        ip = str(ip_address(lookup))
    except ValueError:
        ip = PUBLIC_WEBHOOK_PIN
    sockaddr_port = 0 if port is None else port
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (ip, sockaddr_port),
        )
    ]


@contextmanager
def public_webhook_dns():
    """Patch live and unhooked ``getaddrinfo`` without skipping URL policy.

    ``pin_url_for_ssrf(..., bypass_pins=True)`` calls
    ``ocr_distributed.ssrf._ORIG_GETADDRINFO``. Validation uses
    ``socket.getaddrinfo`` when the pin hook is not the active resolver.
    Patching both keeps either order deterministic. Hostname delivery tests
    must still mock the HTTP transport: this helper does not open sockets,
    but a real opener would connect to ``PUBLIC_WEBHOOK_PIN``.
    """
    from ocr_distributed import ssrf

    resolver = public_webhook_getaddrinfo
    with (
        patch("socket.getaddrinfo", resolver),
        patch.object(ssrf, "_ORIG_GETADDRINFO", resolver),
    ):
        yield
