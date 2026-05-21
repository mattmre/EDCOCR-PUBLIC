"""Property-based testing with Hypothesis.

Tests security-critical string handling using Hypothesis to discover
edge cases that hand-written unit tests would miss.

Targets:
  - _sanitize_path_segment() from ocr_gpu_async.py
  - normalize_pagination() from api/deps.py
  - _get_real_client_ip() from api/limits.py
  - API key format validation (secrets.compare_digest never crashes)
"""

import os
import secrets
import string
import sys
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# 1. _sanitize_path_segment — filesystem-safe string output
# ===========================================================================

# Import the function directly from ocr_gpu_async
from ocr_gpu_async import _sanitize_path_segment  # noqa: E402

# Characters that must never appear in sanitized output
_ILLEGAL_CHARS = set('<>:"|?*\x00')
_CONTROL_CHARS = {chr(c) for c in range(32)}
# Path separators (OS-specific)
_PATH_SEPS = {'/', '\\'}


class TestSanitizePathSegment:
    """Property tests for _sanitize_path_segment()."""

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_never_crashes_on_arbitrary_text(self, text):
        """_sanitize_path_segment must never raise on any input."""
        result = _sanitize_path_segment(text)
        assert isinstance(result, str)

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_no_illegal_chars_in_output(self, text):
        """Output must not contain Windows-illegal or null characters."""
        result = _sanitize_path_segment(text)
        for ch in result:
            assert ch not in _ILLEGAL_CHARS, (
                f"Illegal character {ch!r} (U+{ord(ch):04X}) in sanitized output"
            )

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_no_control_characters_in_output(self, text):
        """Output must not contain control characters (< U+0020)."""
        result = _sanitize_path_segment(text)
        for ch in result:
            assert ord(ch) >= 32, (
                f"Control character U+{ord(ch):04X} in sanitized output"
            )

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_no_null_bytes_in_output(self, text):
        """Output must never contain null bytes."""
        result = _sanitize_path_segment(text)
        assert "\x00" not in result

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_no_leading_trailing_dots_or_spaces(self, text):
        """Output must not start or end with dots or spaces (Windows FAT/NTFS)."""
        result = _sanitize_path_segment(text)
        if result:  # Empty string is allowed
            assert not result.startswith("."), "Output starts with '.'"
            assert not result.startswith(" "), "Output starts with ' '"
            assert not result.endswith("."), "Output ends with '.'"
            assert not result.endswith(" "), "Output ends with ' '"

    @given(st.text(min_size=1))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_idempotent(self, text):
        """Sanitizing an already-sanitized string should produce the same result."""
        first_pass = _sanitize_path_segment(text)
        second_pass = _sanitize_path_segment(first_pass)
        assert first_pass == second_pass, (
            f"Not idempotent: {first_pass!r} -> {second_pass!r}"
        )

    def test_known_dangerous_inputs(self):
        """Explicitly test known dangerous filesystem path segments."""
        dangerous_inputs = [
            "",
            ".",
            "..",
            "...",
            " ",
            "   ",
            "\x00",
            "\x00\x00\x00",
            "CON",  # Windows reserved name (function does not block this, just illegal chars)
            "a" * 1000,  # Very long name
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32",
            'file<with>illegal:chars"and|pipes?and*globs',
            "\r\n\t\x0b\x0c",
            "\u200b\u200c\u200d",  # Zero-width characters
            "\uffff",
            "normal_file.pdf",
        ]
        for inp in dangerous_inputs:
            result = _sanitize_path_segment(inp)
            assert isinstance(result, str)
            # Verify no illegal chars made it through
            for ch in result:
                assert ch not in _ILLEGAL_CHARS
                assert ord(ch) >= 32

    @given(st.binary())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_handles_arbitrary_bytes_decoded_as_text(self, raw_bytes):
        """Bytes decoded with errors='replace' should still sanitize safely."""
        text = raw_bytes.decode("utf-8", errors="replace")
        result = _sanitize_path_segment(text)
        assert isinstance(result, str)
        assert "\x00" not in result


# ===========================================================================
# 2. normalize_pagination — always returns valid (limit, offset)
# ===========================================================================

from api.deps import MAX_PAGE_LIMIT, normalize_pagination  # noqa: E402


class TestNormalizePagination:
    """Property tests for normalize_pagination()."""

    @given(
        limit=st.integers(min_value=1, max_value=MAX_PAGE_LIMIT),
        offset=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_basic_passthrough(self, limit, offset):
        """Without page/per_page, limit and offset pass through unchanged."""
        result_limit, result_offset = normalize_pagination(limit, offset)
        assert result_limit == limit
        assert result_offset == offset

    @given(
        page=st.integers(min_value=1, max_value=1000),
        per_page=st.integers(min_value=1, max_value=MAX_PAGE_LIMIT),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_page_per_page_conversion(self, page, per_page):
        """page/per_page always produces non-negative offset and valid limit."""
        result_limit, result_offset = normalize_pagination(
            50, 0, page=page, per_page=per_page
        )
        assert result_limit == per_page
        assert result_offset == (page - 1) * per_page
        assert result_offset >= 0

    @given(
        limit=st.integers(min_value=1, max_value=MAX_PAGE_LIMIT),
        offset=st.integers(min_value=0, max_value=10_000),
        page=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        per_page=st.one_of(st.none(), st.integers(min_value=1, max_value=MAX_PAGE_LIMIT)),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_output_types_always_valid(self, limit, offset, page, per_page):
        """Output is always a tuple of two non-negative integers."""
        result_limit, result_offset = normalize_pagination(
            limit, offset, page=page, per_page=per_page
        )
        assert isinstance(result_limit, int)
        assert isinstance(result_offset, int)
        assert result_limit >= 1
        assert result_offset >= 0

    @given(
        limit=st.integers(min_value=1, max_value=MAX_PAGE_LIMIT),
        offset=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_non_int_page_ignored(self, limit, offset):
        """Non-integer page/per_page values (e.g. Form sentinels) are ignored."""
        # Simulate Form(...) sentinel objects
        sentinel = MagicMock()
        result_limit, result_offset = normalize_pagination(
            limit, offset, page=sentinel, per_page=sentinel
        )
        # Should fall through to direct limit/offset
        assert result_limit == limit
        assert result_offset == offset


# ===========================================================================
# 3. _get_real_client_ip — never crashes on arbitrary headers
# ===========================================================================

from api.limits import _get_real_client_ip  # noqa: E402


def _make_request_with_headers(headers: dict[str, str]) -> MagicMock:
    """Create a mock Starlette Request with the given headers."""
    request = MagicMock()
    request.headers = headers
    # Provide a fallback client
    client_mock = MagicMock()
    client_mock.host = "127.0.0.1"
    request.client = client_mock
    return request


class TestGetRealClientIP:
    """Property tests for _get_real_client_ip()."""

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_never_crashes_on_xff(self, xff_value):
        """Arbitrary X-Forwarded-For values must never crash."""
        request = _make_request_with_headers({"x-forwarded-for": xff_value})
        result = _get_real_client_ip(request)
        assert isinstance(result, str)

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_never_crashes_on_x_real_ip(self, real_ip_value):
        """Arbitrary X-Real-IP values must never crash."""
        request = _make_request_with_headers({"x-real-ip": real_ip_value})
        result = _get_real_client_ip(request)
        assert isinstance(result, str)

    @given(
        xff=st.text(),
        real_ip=st.text(),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_never_crashes_on_both_headers(self, xff, real_ip):
        """Both headers present simultaneously must never crash."""
        request = _make_request_with_headers({
            "x-forwarded-for": xff,
            "x-real-ip": real_ip,
        })
        result = _get_real_client_ip(request)
        assert isinstance(result, str)

    @given(st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_result_never_empty_string(self, xff_value):
        """Result should never be the empty string (fallback to client.host)."""
        request = _make_request_with_headers({"x-forwarded-for": xff_value})
        result = _get_real_client_ip(request)
        assert len(result) > 0

    @given(
        st.lists(
            st.text(
                alphabet=string.digits + ".:",
                min_size=1,
                max_size=45,
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_xff_chain_returns_first_entry(self, ip_chain):
        """XFF with multiple entries should return the leftmost (first) IP."""
        xff_value = ", ".join(ip_chain)
        request = _make_request_with_headers({"x-forwarded-for": xff_value})
        result = _get_real_client_ip(request)
        expected_first = ip_chain[0].strip()
        if expected_first:
            assert result == expected_first

    def test_no_headers_returns_client_host(self):
        """Without proxy headers, falls back to request.client.host."""
        request = _make_request_with_headers({})
        # The fallback uses slowapi's get_remote_address which checks
        # request.client.host
        result = _get_real_client_ip(request)
        assert isinstance(result, str)

    def test_xff_with_only_whitespace(self):
        """XFF containing only whitespace should fall through to next check."""
        request = _make_request_with_headers({
            "x-forwarded-for": "   ,   ,   ",
        })
        result = _get_real_client_ip(request)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# 4. API key comparison — secrets.compare_digest never crashes
# ===========================================================================


def _safe_compare_digest(a: str, b: str) -> bool:
    """Wrap secrets.compare_digest to handle non-ASCII TypeError gracefully.

    secrets.compare_digest raises TypeError for strings containing characters
    with ordinal > 127.  API keys should be ASCII-only, but a malicious client
    could send non-ASCII headers.  This wrapper catches the TypeError and
    returns False (no match).
    """
    try:
        return secrets.compare_digest(a, b)
    except TypeError:
        return False


class TestAPIKeyComparison:
    """Property tests for API key format handling.

    The auth middleware uses ``secrets.compare_digest`` which raises
    ``TypeError`` for strings with non-ASCII characters (ordinal > 127).
    These tests verify that:
    1. ASCII keys (the valid key space) always compare correctly.
    2. Non-ASCII inputs can be safely handled via a wrapper.
    3. The comparison never produces false positives.
    """

    @given(st.text(alphabet=string.printable))
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_compare_digest_ascii_never_crashes(self, provided_key):
        """secrets.compare_digest should never crash on ASCII text."""
        reference_key = "test-api-key-12345"
        result = secrets.compare_digest(provided_key, reference_key)
        assert isinstance(result, bool)

    @given(
        st.text(alphabet=string.printable),
        st.text(alphabet=string.printable),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_compare_digest_ascii_both_sides(self, a, b):
        """Comparison of two ASCII strings should never crash."""
        result = secrets.compare_digest(a, b)
        assert isinstance(result, bool)
        if a == b:
            assert result is True

    @given(st.text())
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_non_ascii_keys_handled_safely(self, key):
        """Non-ASCII keys must not crash the application.

        secrets.compare_digest raises TypeError for chars > 127.
        A safe wrapper must catch this and return False (no match).
        """
        reference = "valid-api-key-ascii"
        result = _safe_compare_digest(key, reference)
        assert isinstance(result, bool)
        # Non-ASCII key should never match an ASCII reference
        if any(ord(c) > 127 for c in key):
            assert result is False

    @given(st.binary())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_decoded_bytes_handled_safely(self, raw_bytes):
        """Bytes decoded with errors='replace' should compare safely via wrapper."""
        text = raw_bytes.decode("utf-8", errors="replace")
        reference = "valid-api-key"
        result = _safe_compare_digest(text, reference)
        assert isinstance(result, bool)

    @given(st.text(min_size=0, max_size=10000, alphabet=string.printable))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_and_long_ascii_keys(self, key):
        """Empty strings and very long ASCII keys should not crash."""
        reference = "ref-key"
        result = secrets.compare_digest(key, reference)
        assert isinstance(result, bool)

    def test_known_key_matches(self):
        """Verify correct match behavior for known values."""
        key = "my-secret-api-key-abc123"
        assert secrets.compare_digest(key, key) is True
        assert secrets.compare_digest(key, key + "x") is False
        assert secrets.compare_digest(key, "") is False
        assert secrets.compare_digest("", key) is False
        assert secrets.compare_digest("", "") is True

    def test_non_ascii_raises_type_error(self):
        """Document that secrets.compare_digest raises TypeError for non-ASCII.

        This is the actual Python behavior that the auth middleware must
        account for. Non-ASCII API keys from headers would crash the
        middleware without proper handling.
        """
        non_ascii_key = "\x80"
        reference = "valid-key"
        with pytest.raises(TypeError, match="non-ASCII"):
            secrets.compare_digest(non_ascii_key, reference)
