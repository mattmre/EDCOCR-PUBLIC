"""SSRF protection utilities for webhook URL validation.

Shared module used by both the REST API (api/webhooks.py) and the
distributed coordinator (coordinator/jobs/tasks.py).
"""

from __future__ import annotations

import logging
import socket
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from ipaddress import IPv4Address, ip_address, ip_network
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# DNS pin table: hostname (lower) -> IP used for connect. Keeps TLS SNI on the
# original hostname while forcing the socket target to the validated IP
# (anti-rebind). Process-wide; keys are explicit register/unregister.
_PIN_LOCK = threading.RLock()
_PINNED_HOSTS: dict[str, str] = {}
_PIN_REFCOUNTS: dict[str, int] = {}
_ORIG_GETADDRINFO = socket.getaddrinfo
_DNS_PIN_HOOK_INSTALLED = False


def _canonical_host(hostname: str) -> str:
    """Normalize DNS-pin keys without changing the URL used for TLS/SNI."""
    return str(hostname).strip().lower().rstrip(".")


def _pinned_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """socket.getaddrinfo wrapper that forces registered host → pinned IP."""
    lookup = host
    if isinstance(host, bytes):
        try:
            lookup = host.decode("idna")
        except Exception:
            lookup = host.decode("utf-8", errors="replace")
    mapped = None
    with _PIN_LOCK:
        mapped = _PINNED_HOSTS.get(_canonical_host(str(lookup)))
    if mapped:
        host = mapped
    return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)


def install_dns_pin_hook() -> None:
    """Install process-wide getaddrinfo hook once (idempotent)."""
    global _DNS_PIN_HOOK_INSTALLED
    with _PIN_LOCK:
        if _DNS_PIN_HOOK_INSTALLED:
            return
        socket.getaddrinfo = _pinned_getaddrinfo  # type: ignore[assignment]
        _DNS_PIN_HOOK_INSTALLED = True


def register_host_pin(hostname: str, ip: str) -> None:
    """Force *hostname* resolutions to *ip* for the process lifetime of the pin."""
    if not hostname or not ip:
        return
    install_dns_pin_hook()
    with _PIN_LOCK:
        key = _canonical_host(hostname)
        active_ip = _PINNED_HOSTS.get(key)
        if active_ip and active_ip != ip:
            raise ValueError(
                f"Hostname {hostname!r} is already pinned to a different IP"
            )
        _PINNED_HOSTS[key] = ip
        _PIN_REFCOUNTS[key] = _PIN_REFCOUNTS.get(key, 0) + 1


def unregister_host_pin(hostname: str) -> None:
    """Drop a previously registered host pin (no-op if missing)."""
    if not hostname:
        return
    with _PIN_LOCK:
        key = _canonical_host(hostname)
        if key not in _PINNED_HOSTS:
            return
        refcount = _PIN_REFCOUNTS.get(key, 1)
        if refcount > 1:
            _PIN_REFCOUNTS[key] = refcount - 1
            return
        _PIN_REFCOUNTS.pop(key, None)
        _PINNED_HOSTS.pop(key, None)


def get_registered_pin(hostname: str) -> str | None:
    """Return pinned IP for *hostname*, or None."""
    if not hostname:
        return None
    with _PIN_LOCK:
        return _PINNED_HOSTS.get(_canonical_host(hostname))


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP handler that blocks redirects to prevent SSRF bypass."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            f"Redirect blocked (SSRF protection): {newurl}",
            headers,
            fp,
        )


# Module-level opener that blocks redirects and ignores ambient proxy
# environment variables (SSRF protection).  A proxy would receive the
# hostname instead of the pinned socket destination and could resolve a
# rebinding target outside this process's DNS pin hook.
safe_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    NoRedirectHandler,
)


_NAT64_WELL_KNOWN = ip_network("64:ff9b::/96")
_SIXTO4_PREFIX = ip_network("2002::/16")
_TEREDO_PREFIX = ip_network("2001:0::/32")


def _teredo_inner_ipv4s(addr):
    """Return (server, client) IPv4 from an RFC 4380 Teredo address."""
    if getattr(addr, "version", None) != 6 or addr not in _TEREDO_PREFIX:
        return ()
    server = IPv4Address(addr.packed[4:8])
    client = IPv4Address(int.from_bytes(addr.packed[12:16], "big") ^ 0xFFFFFFFF)
    return (server, client)


def _embedded_ipv4(addr):
    """Return the inner IPv4 address for mapped / compatible / NAT64 / 6to4 / Teredo."""
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        return mapped
    compatible = getattr(addr, "ipv4_compatible", None)
    if compatible is not None:
        return compatible
    if getattr(addr, "version", None) != 6:
        return None
    if addr in _NAT64_WELL_KNOWN:
        return IPv4Address(addr.packed[12:])
    if addr in _SIXTO4_PREFIX:
        return IPv4Address(addr.packed[2:6])
    # RFC 5214 ISATAP IID: 0000:5EFE:x.x.x.x or 0200:5EFE:x.x.x.x
    iid = addr.packed[8:12]
    if iid in (b"\x00\x00\x5e\xfe", b"\x02\x00\x5e\xfe"):
        return IPv4Address(addr.packed[12:16])
    teredo = _teredo_inner_ipv4s(addr)
    if teredo:
        return teredo[1]
    return None


def _raw_addr_blocked(addr) -> bool:
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_reserved
        or addr.is_link_local
        or addr.is_multicast
        or bool(getattr(addr, "is_site_local", False))
        or not addr.is_global
    )


def _addr_blocked(addr) -> bool:
    """True when *addr* is private, loopback, reserved, link-local, or site-local.

    Python's ``is_private`` does **not** cover link-local (``169.254.0.0/16``,
    ``fe80::/10``). Cloud metadata endpoints often live on link-local, so
    webhook SSRF must treat them as blocked.

    IPv4-mapped IPv6 (``::ffff:169.254.169.254``), deprecated IPv4-compatible
    (``::169.254.169.254``), NAT64 well-known prefix (``64:ff9b::/96``),
    6to4 (``2002::/16``), Teredo (``2001:0::/32``), and ISATAP IIDs
    (``…:5efe:`` / ``…:200:5efe:``) must be judged as the inner IPv4 address;
    otherwise IMDS/RFC1918 bypasses the v4 checks.
    Teredo is blocked when *either* the server or XOR-obfuscated client IPv4
    is blocked.
    """
    teredo = _teredo_inner_ipv4s(addr)
    if teredo:
        return any(_raw_addr_blocked(inner) for inner in teredo)
    embedded = _embedded_ipv4(addr)
    if embedded is not None:
        addr = embedded
    return _raw_addr_blocked(addr)


def resolve_host_ips(hostname: str, *, bypass_pins: bool = False) -> list[str]:
    """Resolve *hostname* to a de-duplicated list of IP address strings.

    ``bypass_pins`` is used for a fresh security decision.  The process-wide
    getaddrinfo hook intentionally returns an already registered pin, but a
    request that is about to register a pin must compare against unhooked DNS
    rather than treating the old pin as fresh validation.

    Returns an empty list when DNS fails (caller decides fail-closed vs pin-skip).
    """
    hostname = _canonical_host(hostname)
    if not hostname:
        return []
    try:
        # Literal IP — return as-is without DNS.
        return [str(ip_address(hostname))]
    except ValueError:
        pass
    ips: list[str] = []
    try:
        resolver = _ORIG_GETADDRINFO if bypass_pins else socket.getaddrinfo
        resolved = resolver(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except (socket.gaierror, OSError):
        return []
    for _family, _type, _proto, _canonname, sockaddr in resolved:
        ip = sockaddr[0]
        if ip not in ips:
            ips.append(ip)
    return ips


def is_private_ip(hostname: str) -> bool:
    """Check if hostname is/resolves to a private/loopback/link-local address."""
    ips = resolve_host_ips(hostname)
    if not ips:
        # Cannot resolve -- treat as potentially private for safety
        return True
    for ip in ips:
        if _addr_blocked(ip_address(ip)):
            return True
    return False


def _format_netloc(hostname, port=None, username=None, password=None):
    """Build a URL netloc, bracketing IPv6 literals."""
    try:
        addr = ip_address(hostname)
        host_part = f"[{addr}]" if addr.version == 6 else str(addr)
    except ValueError:
        host_part = hostname
    if port is not None:
        host_part = f"{host_part}:{port}"
    if username is not None:
        userinfo = username
        if password is not None:
            userinfo = f"{userinfo}:{password}"
        return f"{userinfo}@{host_part}"
    return host_part


def pin_url_for_ssrf(
    url: str,
    *,
    allow_http: bool = False,
    allow_private: bool = False,
    rewrite_url_to_ip: bool = False,
    bypass_pins: bool = False,
) -> tuple[str, str, str]:
    """Validate *url*, resolve DNS once, and return a pin for connect-time.

    DNS rebinding is closed by registering the resolved IP so connect uses that
    IP while the URL hostname (and TLS SNI / cert verify) stay on the original
    name. Call ``register_host_pin(hostname, pinned_ip)`` before opening the
    HTTP client.

    When *rewrite_url_to_ip* is True, the returned URL host is rewritten to the
    pin and the caller must set ``Host: original_hostname`` (HTTP-only use).

    Returns:
        ``(url_or_rewritten, original_hostname, pinned_ip)``

        *pinned_ip* is never empty: empty resolve is fail-closed even when
        ``allow_private=True`` so a later NXDOMAIN→IMDS rebind cannot skip
        the pin.

    Raises:
        ValueError: scheme/length/private policy failure, or DNS miss.
    """
    url = validate_webhook_url(url, allow_http=allow_http, allow_private=allow_private)
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError("Webhook URL is malformed.") from exc
    hostname = _canonical_host(parsed.hostname or "")
    if not hostname:
        raise ValueError("Webhook URL must include a hostname.")

    ips = resolve_host_ips(hostname, bypass_pins=bypass_pins)
    if not ips:
        raise ValueError(
            "Cannot resolve hostname for SSRF pin; refusing unpinned endpoint."
        )

    # Fail closed if any answer is blocked under the active policy (same as
    # is_private_ip aggregation), then pin the first allowed IP.
    for ip in ips:
        blocked = _addr_blocked(ip_address(ip))
        if blocked and not allow_private:
            raise ValueError(
                "Webhook URL must not point to private or reserved IP addresses."
            )

    pinned_ip = ips[0]
    try:
        # Hostname already a literal IP — pin is the IP itself.
        ip_address(hostname)
        return url, hostname, pinned_ip
    except ValueError:
        pass

    if not rewrite_url_to_ip:
        # Hostname URL + getaddrinfo pin keeps HTTPS SNI / cert verify correct.
        return url, hostname, pinned_ip

    netloc = _format_netloc(
        pinned_ip,
        parsed.port,
        parsed.username,
        parsed.password,
    )
    pinned = parsed._replace(netloc=netloc).geturl()
    return pinned, hostname, pinned_ip


def host_header_for_pin(original_hostname: str, port: int | None) -> str:
    """Build the Host header for a pinned (IP-rewritten) request."""
    if port is None:
        return original_hostname
    return f"{original_hostname}:{port}"


@contextmanager
def pinned_url_for_request(
    url: str,
    *,
    allow_http: bool = False,
    allow_private: bool = False,
):
    """Yield a URL whose hostname is pinned for the active request.

    This is the delivery-side companion to :func:`pin_url_for_ssrf`.  It
    resolves and validates immediately before opening the socket, installs the
    process-local hostname pin, and removes that temporary pin afterwards.
    Callers keep the original hostname in the URL so HTTPS certificate/SNI
    behavior remains correct.

    A conflicting active pin for the same hostname is refused rather than
    overwritten.  Overwriting a concurrent request's pin would reintroduce a
    DNS-rebinding race.  Pin registrations are reference-counted so a short
    webhook request cannot remove a longer-lived VLM gateway pin for the same
    host.
    """
    pinned_url, original_host, pinned_ip = pin_url_for_ssrf(
        url,
        allow_http=allow_http,
        allow_private=allow_private,
        bypass_pins=True,
    )
    if not pinned_ip or original_host == pinned_ip:
        yield pinned_url
        return

    with _PIN_LOCK:
        active_pin = _PINNED_HOSTS.get(original_host)
        if active_pin and active_pin != pinned_ip:
            raise ValueError(
                f"Hostname {original_host!r} is already pinned to a different IP"
            )
        register_host_pin(original_host, pinned_ip)
    try:
        yield pinned_url
    finally:
        unregister_host_pin(original_host)


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

    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("Webhook URL must not contain control characters.")
    url = url.strip()
    if len(url) > 2048:
        raise ValueError("Webhook URL exceeds maximum length of 2048 characters.")

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError("Webhook URL is malformed.") from exc

    if not parsed.scheme:
        raise ValueError("Webhook URL must include a scheme (https://).")

    if parsed.scheme == "http" and not allow_http:
        raise ValueError(
            "Webhook URL must use HTTPS. Set WEBHOOK_ALLOW_HTTP=true for development."
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL has unsupported scheme: {parsed.scheme}")

    try:
        parsed_hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("Webhook URL has a malformed hostname.") from exc
    if not parsed_hostname:
        raise ValueError("Webhook URL must include a hostname.")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "Webhook URL must not contain username or password information."
        )
    if parsed.fragment:
        raise ValueError("Webhook URL must not contain a fragment.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Webhook URL contains an invalid port.") from exc

    hostname = _canonical_host(parsed_hostname or "")
    if not hostname:
        raise ValueError("Webhook URL must include a hostname.")
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
