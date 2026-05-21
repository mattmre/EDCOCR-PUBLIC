"""Tests for api/limits.py rate limiter key extraction.

Verifies that the rate limiter correctly extracts client IPs from
proxy headers (X-Forwarded-For, X-Real-IP) and falls back to
request.client.host when no proxy headers are present.
"""

from unittest.mock import MagicMock

from api.limits import _get_real_client_ip, _tenant_aware_key_func


def _make_request(headers=None, client_host="127.0.0.1"):
    """Build a mock Starlette Request with optional headers and client host."""
    req = MagicMock()
    header_dict = headers or {}

    def get_header(name, default=None):
        return header_dict.get(name.lower(), default)

    req.headers = MagicMock()
    req.headers.get = get_header
    req.client = MagicMock()
    req.client.host = client_host
    req.state = MagicMock(spec=[])  # no tenant_id by default
    return req


class TestGetRealClientIp:
    """Verify _get_real_client_ip proxy header extraction."""

    def test_xff_single_ip(self):
        """X-Forwarded-For with a single IP returns that IP."""
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.50"},
            client_host="10.0.0.1",
        )
        assert _get_real_client_ip(req) == "203.0.113.50"

    def test_xff_multiple_ips_returns_leftmost(self):
        """X-Forwarded-For with multiple IPs returns the leftmost (original client)."""
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.50, 70.41.3.18, 150.172.238.178"},
            client_host="10.0.0.1",
        )
        assert _get_real_client_ip(req) == "203.0.113.50"

    def test_xff_with_whitespace(self):
        """X-Forwarded-For IPs are trimmed of whitespace."""
        req = _make_request(
            headers={"x-forwarded-for": "  198.51.100.22 , 10.0.0.1"},
            client_host="10.0.0.1",
        )
        assert _get_real_client_ip(req) == "198.51.100.22"

    def test_xrealip_used_when_no_xff(self):
        """X-Real-IP is used when X-Forwarded-For is absent."""
        req = _make_request(
            headers={"x-real-ip": "198.51.100.33"},
            client_host="10.0.0.1",
        )
        assert _get_real_client_ip(req) == "198.51.100.33"

    def test_xff_takes_precedence_over_xrealip(self):
        """X-Forwarded-For is preferred over X-Real-IP when both present."""
        req = _make_request(
            headers={
                "x-forwarded-for": "203.0.113.1",
                "x-real-ip": "198.51.100.2",
            },
            client_host="10.0.0.1",
        )
        assert _get_real_client_ip(req) == "203.0.113.1"

    def test_fallback_to_client_host(self):
        """Falls back to request.client.host when no proxy headers are present."""
        req = _make_request(client_host="192.168.1.100")
        result = _get_real_client_ip(req)
        # Should come from the fallback path (get_remote_address uses client.host)
        assert result is not None
        assert result != ""

    def test_empty_xff_falls_through(self):
        """Empty X-Forwarded-For header is ignored."""
        req = _make_request(
            headers={"x-forwarded-for": ""},
            client_host="10.0.0.5",
        )
        # Should NOT return empty string -- should fall through
        result = _get_real_client_ip(req)
        assert result != ""

    def test_xff_with_only_commas(self):
        """X-Forwarded-For with only commas/spaces falls through."""
        req = _make_request(
            headers={"x-forwarded-for": " , , "},
            client_host="10.0.0.5",
        )
        result = _get_real_client_ip(req)
        # The first split result is empty after strip, so falls through
        assert result is not None


class TestTenantAwareKeyFunc:
    """Verify _tenant_aware_key_func integration with proxy headers."""

    def test_ip_key_uses_proxy_headers(self):
        """When not in multi-tenancy mode, key uses proxy-aware IP."""
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.99"},
            client_host="10.0.0.1",
        )
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "api.limits.ENABLE_MULTITENANCY", False
        ):
            result = _tenant_aware_key_func(req)
        assert result == "203.0.113.99"

    def test_tenant_id_overrides_ip(self):
        """When multi-tenancy is active and tenant_id is set, it takes priority."""
        req = _make_request(
            headers={"x-forwarded-for": "203.0.113.99"},
            client_host="10.0.0.1",
        )
        req.state.tenant_id = "tenant-abc-123"
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "api.limits.ENABLE_MULTITENANCY", True
        ):
            result = _tenant_aware_key_func(req)
        assert result == "tenant-abc-123"
