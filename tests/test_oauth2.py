"""Tests for OAuth2 JWT validation, JWKS caching, and OIDC discovery.

Covers:
- TestJWTValidation: valid token, expired token, invalid signature, wrong audience,
  wrong issuer, no keys, no audience check
- TestJWKSCache: cache hit, cache miss (fetch), cache expiry, invalidate, graceful
  degradation, empty on initial failure
- TestOIDCDiscovery: JWKS URI derived from issuer, missing config raises, explicit
  JWKS URI preferred, unreachable OIDC issuer
- TestRoleExtraction: admin/operator/viewer from list, string role, missing claim,
  unknown value, priority ordering, custom claim path, custom role mapping, numeric
  claim, empty list claim
- TestOAuth2Disabled: OAUTH2_ENABLED=false skips JWT validation entirely
- TestLazyImports: _ensure_jwt and _ensure_httpx raise ImportError gracefully
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers -- generate JWTs for testing using PyJWT
# ---------------------------------------------------------------------------

try:
    import jwt as pyjwt
    from jwt.algorithms import RSAAlgorithm

    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False

if _HAS_JWT:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate a deterministic RSA key pair for tests
    _TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _TEST_PUBLIC_KEY = _TEST_PRIVATE_KEY.public_key()

    _TEST_PRIVATE_PEM = _TEST_PRIVATE_KEY.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    _TEST_PUBLIC_PEM = _TEST_PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    # Build a JWKS-format key from the public key using PyJWT
    _TEST_JWK = json.loads(RSAAlgorithm.to_jwk(_TEST_PUBLIC_KEY))
    _TEST_JWK["kid"] = "test-key-1"
    _TEST_JWK["use"] = "sig"
    _TEST_JWK["alg"] = "RS256"

    _TEST_JWKS = {"keys": [_TEST_JWK]}
else:
    _TEST_PRIVATE_PEM = ""
    _TEST_JWKS = {"keys": []}

pytestmark = pytest.mark.skipif(not _HAS_JWT, reason="PyJWT not installed")


def _make_token(
    claims: dict | None = None,
    exp_offset: int = 3600,
    key: str | None = None,
    algorithm: str = "RS256",
) -> str:
    """Create a signed JWT for testing."""
    now = int(time.time())
    payload = {
        "sub": "test-user",
        "iss": "https://test-issuer.example.com",
        "aud": "test-audience",
        "iat": now,
        "nbf": now,
        "exp": now + exp_offset,
        "email": "test@example.com",
        "name": "Test User",
        "roles": ["operator"],
    }
    if claims:
        payload.update(claims)
    return pyjwt.encode(payload, key or _TEST_PRIVATE_PEM, algorithm=algorithm)


def _make_jwks_cache(keys=None, fetched_at=None, ttl=3600):
    """Create a pre-populated JWKSCache instance for testing."""
    from api.oauth2 import JWKSCache

    cache = JWKSCache.__new__(JWKSCache)
    cache._jwks_uri = "https://test-issuer.example.com/.well-known/jwks.json"
    cache._keys = keys if keys is not None else _TEST_JWKS["keys"]
    cache._fetched_at = fetched_at if fetched_at is not None else time.time()
    cache._ttl = ttl
    cache._lock = threading.Lock()
    return cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_oauth2_module():
    """Reset module-level singletons between tests."""
    from api.oauth2 import reset_jwks_cache

    reset_jwks_cache()
    yield
    reset_jwks_cache()


@pytest.fixture()
def mock_oauth2_env():
    """Patch OAuth2 config env vars for testing."""
    with patch("api.oauth2_config.OAUTH2_ENABLED", True), \
         patch("api.oauth2_config.OAUTH2_ISSUER", "https://test-issuer.example.com"), \
         patch("api.oauth2_config.OAUTH2_AUDIENCE", "test-audience"), \
         patch("api.oauth2_config.OAUTH2_JWKS_URI", "https://test-issuer.example.com/.well-known/jwks.json"), \
         patch("api.oauth2_config.OAUTH2_ALGORITHMS", ["RS256"]), \
         patch("api.oauth2_config.OAUTH2_ROLE_CLAIM", "roles"), \
         patch("api.oauth2_config.OAUTH2_ADMIN_ROLE", "admin"), \
         patch("api.oauth2_config.OAUTH2_OPERATOR_ROLE", "operator"), \
         patch("api.oauth2_config.OAUTH2_VIEWER_ROLE", "viewer"), \
         patch("api.oauth2_config.OAUTH2_DEFAULT_ROLE", "viewer"), \
         patch("api.oauth2_config.APIKEY_ROLE", "admin"):
        yield


# ---------------------------------------------------------------------------
# JWKSCache tests
# ---------------------------------------------------------------------------

class TestJWKSCache:
    def test_cache_fetches_keys_on_first_call(self, mock_oauth2_env):
        """Cache fetches JWKS on first get_keys() call."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _TEST_JWKS
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            keys = cache.get_keys()

        assert len(keys) == 1
        assert keys[0]["kid"] == "test-key-1"
        mock_httpx.get.assert_called_once()

    def test_cache_returns_cached_keys_on_subsequent_calls(self, mock_oauth2_env):
        """Subsequent calls return cached keys without re-fetching."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _TEST_JWKS
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            cache.get_keys()
            cache.get_keys()
            cache.get_keys()

        # Only fetched once due to caching
        assert mock_httpx.get.call_count == 1

    def test_cache_refreshes_after_ttl_expiry(self, mock_oauth2_env):
        """Cache refreshes when TTL has elapsed."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _TEST_JWKS
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json", ttl=1)
            cache.get_keys()

            # Simulate TTL expiry
            cache._fetched_at = time.time() - 2
            cache.get_keys()

        assert mock_httpx.get.call_count == 2

    def test_cache_invalidate_forces_refresh(self, mock_oauth2_env):
        """invalidate() causes next get_keys() to re-fetch."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = _TEST_JWKS
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            cache.get_keys()
            cache.invalidate()
            cache.get_keys()

        assert mock_httpx.get.call_count == 2

    def test_cache_graceful_degradation_on_fetch_error(self, mock_oauth2_env):
        """Cache returns stale keys when fetch fails."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_response = MagicMock()
                mock_response.json.return_value = _TEST_JWKS
                mock_response.raise_for_status.return_value = None
                return mock_response
            raise ConnectionError("Network unreachable")

        mock_httpx.get.side_effect = side_effect

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            keys1 = cache.get_keys()
            cache._fetched_at = 0  # Force refresh
            keys2 = cache.get_keys()

        # Should still return stale keys
        assert len(keys1) == 1
        assert len(keys2) == 1

    def test_cache_empty_on_initial_failure(self, mock_oauth2_env):
        """Cache returns empty list if first fetch fails."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = ConnectionError("Network unreachable")

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            keys = cache.get_keys()

        assert keys == []

    def test_cache_jwks_uri_property(self, mock_oauth2_env):
        """jwks_uri property returns the configured URI."""
        from api.oauth2 import JWKSCache

        cache = JWKSCache("https://example.com/.well-known/jwks.json")
        assert cache.jwks_uri == "https://example.com/.well-known/jwks.json"

    def test_cache_handles_empty_keys_in_response(self, mock_oauth2_env):
        """Cache handles response with no keys array."""
        from api.oauth2 import JWKSCache

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # No "keys" field
        mock_response.raise_for_status.return_value = None
        mock_httpx.get.return_value = mock_response

        with patch("api.oauth2._httpx", mock_httpx):
            cache = JWKSCache("https://test-issuer.example.com/.well-known/jwks.json")
            keys = cache.get_keys()

        assert keys == []


# ---------------------------------------------------------------------------
# JWT validation tests
# ---------------------------------------------------------------------------

class TestJWTValidation:
    def test_valid_token_decoded_successfully(self, mock_oauth2_env):
        """Valid JWT is decoded successfully."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())
        token = _make_token()
        payload = validate_jwt(token)

        assert payload["sub"] == "test-user"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["operator"]

    def test_expired_token_raises_valueerror(self, mock_oauth2_env):
        """Expired JWT raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())
        token = _make_token(exp_offset=-3600)  # Already expired

        with pytest.raises(ValueError, match="expired"):
            validate_jwt(token)

    def test_bad_signature_raises_valueerror(self, mock_oauth2_env):
        """JWT signed with wrong key raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())

        # Generate a different RSA key
        wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        wrong_pem = wrong_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        token = _make_token(key=wrong_pem)

        with pytest.raises(ValueError, match="validation failed"):
            validate_jwt(token)

    def test_wrong_audience_raises_valueerror(self, mock_oauth2_env):
        """JWT with wrong audience raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())
        token = _make_token(claims={"aud": "wrong-audience"})

        with pytest.raises(ValueError, match="claims"):
            validate_jwt(token)

    def test_wrong_issuer_raises_valueerror(self, mock_oauth2_env):
        """JWT with wrong issuer raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())
        token = _make_token(claims={"iss": "https://evil-issuer.example.com"})

        with pytest.raises(ValueError, match="claims"):
            validate_jwt(token)

    def test_no_keys_available_raises_valueerror(self, mock_oauth2_env):
        """No JWKS keys available raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache(keys=[]))
        token = _make_token()

        with pytest.raises(ValueError, match="No JWKS keys"):
            validate_jwt(token)

    def test_valid_token_without_audience_check(self, mock_oauth2_env):
        """JWT validates when OAUTH2_AUDIENCE is empty (no aud check)."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())

        with patch("api.oauth2_config.OAUTH2_AUDIENCE", ""):
            token = _make_token(claims={"aud": "anything"})
            payload = validate_jwt(token)
            assert payload["sub"] == "test-user"

    def test_token_with_all_claims_present(self, mock_oauth2_env):
        """Token with name, email, and custom claims decodes fully."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())
        token = _make_token(claims={
            "name": "Jane Admin",
            "email": "jane@corp.example.com",
            "department": "engineering",
        })
        payload = validate_jwt(token)

        assert payload["name"] == "Jane Admin"
        assert payload["email"] == "jane@corp.example.com"
        assert payload["department"] == "engineering"

    def test_completely_malformed_token_raises(self, mock_oauth2_env):
        """Completely malformed token string raises ValueError."""
        from api.oauth2 import set_jwks_cache, validate_jwt

        set_jwks_cache(_make_jwks_cache())

        with pytest.raises(ValueError, match="validation failed"):
            validate_jwt("not-a-jwt-at-all")


# ---------------------------------------------------------------------------
# OIDC discovery / JWKS singleton tests
# ---------------------------------------------------------------------------

class TestOIDCDiscovery:
    def test_jwks_uri_derived_from_issuer(self):
        """When OAUTH2_JWKS_URI is empty, JWKS URI is derived from OAUTH2_ISSUER."""
        from api.oauth2 import _get_jwks_cache, reset_jwks_cache

        reset_jwks_cache()

        with patch("api.oauth2_config.OAUTH2_ISSUER", "https://auth.example.com/realm"), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", ""):
            cache = _get_jwks_cache()
            assert cache.jwks_uri == "https://auth.example.com/realm/.well-known/jwks.json"

    def test_explicit_jwks_uri_preferred(self):
        """Explicit OAUTH2_JWKS_URI takes priority over derived URI."""
        from api.oauth2 import _get_jwks_cache, reset_jwks_cache

        reset_jwks_cache()

        with patch("api.oauth2_config.OAUTH2_ISSUER", "https://auth.example.com"), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", "https://custom.example.com/keys"):
            cache = _get_jwks_cache()
            assert cache.jwks_uri == "https://custom.example.com/keys"

    def test_missing_issuer_and_jwks_uri_raises(self):
        """RuntimeError is raised when both OAUTH2_ISSUER and OAUTH2_JWKS_URI are empty."""
        from api.oauth2 import _get_jwks_cache, reset_jwks_cache

        reset_jwks_cache()

        with patch("api.oauth2_config.OAUTH2_ISSUER", ""), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", ""):
            with pytest.raises(RuntimeError, match="OAUTH2_JWKS_URI or OAUTH2_ISSUER"):
                _get_jwks_cache()

    def test_issuer_trailing_slash_stripped(self):
        """Trailing slash on issuer is stripped before deriving JWKS URI."""
        from api.oauth2 import _get_jwks_cache, reset_jwks_cache

        reset_jwks_cache()

        with patch("api.oauth2_config.OAUTH2_ISSUER", "https://auth.example.com/"), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", ""):
            cache = _get_jwks_cache()
            assert cache.jwks_uri == "https://auth.example.com/.well-known/jwks.json"

    def test_singleton_reuses_cache_instance(self):
        """_get_jwks_cache() returns the same instance on repeated calls."""
        from api.oauth2 import _get_jwks_cache, reset_jwks_cache

        reset_jwks_cache()

        with patch("api.oauth2_config.OAUTH2_ISSUER", "https://auth.example.com"), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", "https://auth.example.com/jwks"):
            cache1 = _get_jwks_cache()
            cache2 = _get_jwks_cache()
            assert cache1 is cache2


# ---------------------------------------------------------------------------
# Role extraction tests
# ---------------------------------------------------------------------------

class TestRoleExtraction:
    def test_admin_role_from_list(self, mock_oauth2_env):
        """Extract admin role from list claim."""
        from api.oauth2 import extract_role

        claims = {"roles": ["admin", "viewer"]}
        assert extract_role(claims) == "admin"

    def test_operator_role_from_list(self, mock_oauth2_env):
        """Extract operator role from list claim."""
        from api.oauth2 import extract_role

        claims = {"roles": ["operator"]}
        assert extract_role(claims) == "operator"

    def test_viewer_role_from_list(self, mock_oauth2_env):
        """Extract viewer role from list claim."""
        from api.oauth2 import extract_role

        claims = {"roles": ["viewer"]}
        assert extract_role(claims) == "viewer"

    def test_string_role_value(self, mock_oauth2_env):
        """Extract role from string claim."""
        from api.oauth2 import extract_role

        claims = {"roles": "operator"}
        assert extract_role(claims) == "operator"

    def test_default_role_on_missing_claim(self, mock_oauth2_env):
        """Default role returned when claim is missing."""
        from api.oauth2 import extract_role

        claims = {}
        assert extract_role(claims) == "viewer"

    def test_default_role_on_unknown_string_value(self, mock_oauth2_env):
        """Default role returned for unrecognized string value."""
        from api.oauth2 import extract_role

        claims = {"roles": "superuser"}
        assert extract_role(claims) == "viewer"

    def test_default_role_on_empty_list(self, mock_oauth2_env):
        """Default role returned for empty list claim."""
        from api.oauth2 import extract_role

        claims = {"roles": []}
        assert extract_role(claims) == "viewer"

    def test_default_role_on_unknown_list_values(self, mock_oauth2_env):
        """Default role returned when list contains no matching roles."""
        from api.oauth2 import extract_role

        claims = {"roles": ["superuser", "manager"]}
        assert extract_role(claims) == "viewer"

    def test_priority_admin_over_operator_in_list(self, mock_oauth2_env):
        """Admin role takes priority over operator in list."""
        from api.oauth2 import extract_role

        claims = {"roles": ["operator", "admin"]}
        assert extract_role(claims) == "admin"

    def test_priority_operator_over_viewer_in_list(self, mock_oauth2_env):
        """Operator role takes priority over viewer in list."""
        from api.oauth2 import extract_role

        claims = {"roles": ["viewer", "operator"]}
        assert extract_role(claims) == "operator"

    def test_custom_role_claim_path(self, mock_oauth2_env):
        """Custom role claim path is used."""
        from api.oauth2 import extract_role

        with patch("api.oauth2_config.OAUTH2_ROLE_CLAIM", "realm_access"):
            claims = {"realm_access": "admin"}
            assert extract_role(claims) == "admin"

    def test_custom_role_mapping(self, mock_oauth2_env):
        """Custom external role name mappings work correctly."""
        from api.oauth2 import extract_role

        with patch("api.oauth2_config.OAUTH2_ADMIN_ROLE", "OcrAdmin"):
            claims = {"roles": ["OcrAdmin"]}
            assert extract_role(claims) == "admin"

    def test_numeric_claim_returns_default(self, mock_oauth2_env):
        """Numeric role claim value returns default role."""
        from api.oauth2 import extract_role

        claims = {"roles": 42}
        assert extract_role(claims) == "viewer"

    def test_none_claim_value_returns_default(self, mock_oauth2_env):
        """None role claim value returns default role (same as missing)."""
        from api.oauth2 import extract_role

        claims = {"roles": None}
        assert extract_role(claims) == "viewer"


# ---------------------------------------------------------------------------
# OAuth2 disabled tests
# ---------------------------------------------------------------------------

class TestOAuth2Disabled:
    def test_try_bearer_auth_returns_none_when_disabled(self):
        """_try_bearer_auth returns None when OAUTH2_ENABLED is False."""
        from api.auth import _try_bearer_auth

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer some-token"}

        with patch("api.oauth2_config.OAUTH2_ENABLED", False):
            result = _try_bearer_auth(mock_request)

        assert result is None

    def test_try_bearer_auth_returns_none_without_bearer_header(self):
        """_try_bearer_auth returns None when no Bearer header is present."""
        from api.auth import _try_bearer_auth

        mock_request = MagicMock()
        mock_request.headers = {}

        with patch("api.oauth2_config.OAUTH2_ENABLED", True):
            result = _try_bearer_auth(mock_request)

        assert result is None

    def test_try_bearer_auth_returns_none_for_empty_bearer(self):
        """_try_bearer_auth returns None when Bearer value is empty."""
        from api.auth import _try_bearer_auth

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer "}

        with patch("api.oauth2_config.OAUTH2_ENABLED", True):
            result = _try_bearer_auth(mock_request)

        assert result is None

    def test_try_bearer_auth_returns_none_on_jwt_error(self, mock_oauth2_env):
        """_try_bearer_auth returns None on JWT validation failure."""
        from api.auth import _try_bearer_auth

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer invalid.jwt.token"}

        with patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.oauth2.validate_jwt", side_effect=ValueError("bad token")):
            result = _try_bearer_auth(mock_request)

        assert result is None

    def test_try_bearer_auth_returns_identity_on_success(self, mock_oauth2_env):
        """_try_bearer_auth returns AuthIdentity on successful JWT validation."""
        from api.auth import _try_bearer_auth

        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer valid-token"}

        fake_claims = {
            "sub": "user-abc",
            "email": "user@example.com",
            "name": "Test User",
            "roles": ["operator"],
        }

        with patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.oauth2.validate_jwt", return_value=fake_claims), \
             patch("api.oauth2.extract_role", return_value="operator"):
            result = _try_bearer_auth(mock_request)

        assert result is not None
        assert result.subject == "user-abc"
        assert result.role == "operator"
        assert result.auth_method == "oauth2"
        assert result.email == "user@example.com"


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------

class TestLazyImports:
    def test_ensure_jwt_raises_on_missing_import(self):
        """_ensure_jwt raises ImportError with helpful message when PyJWT is missing."""
        import api.oauth2 as oauth2_mod

        # Save originals and simulate missing jwt
        orig_jwt = oauth2_mod._pyjwt
        orig_exc = oauth2_mod._pyjwt_exceptions
        try:
            oauth2_mod._pyjwt = None
            oauth2_mod._pyjwt_exceptions = None
            with patch.dict("sys.modules", {"jwt": None, "jwt.exceptions": None}):
                with pytest.raises(ImportError, match="PyJWT"):
                    oauth2_mod._ensure_jwt()
        finally:
            oauth2_mod._pyjwt = orig_jwt
            oauth2_mod._pyjwt_exceptions = orig_exc

    def test_ensure_httpx_raises_on_missing_import(self):
        """_ensure_httpx raises ImportError with helpful message when httpx is missing."""
        import api.oauth2 as oauth2_mod

        orig_httpx = oauth2_mod._httpx
        try:
            oauth2_mod._httpx = None
            with patch.dict("sys.modules", {"httpx": None}):
                with pytest.raises(ImportError, match="httpx"):
                    oauth2_mod._ensure_httpx()
        finally:
            oauth2_mod._httpx = orig_httpx


# ---------------------------------------------------------------------------
# set_jwks_cache / reset_jwks_cache tests
# ---------------------------------------------------------------------------

class TestJWKSCacheManagement:
    def test_set_jwks_cache_replaces_singleton(self, mock_oauth2_env):
        """set_jwks_cache replaces the global cache instance."""
        from api.oauth2 import _get_jwks_cache, set_jwks_cache

        custom_cache = _make_jwks_cache()
        set_jwks_cache(custom_cache)

        retrieved = _get_jwks_cache()
        assert retrieved is custom_cache

    def test_reset_jwks_cache_clears_singleton(self, mock_oauth2_env):
        """reset_jwks_cache clears the global cache."""
        from api.oauth2 import reset_jwks_cache, set_jwks_cache

        set_jwks_cache(_make_jwks_cache())
        reset_jwks_cache()

        # Next call to _get_jwks_cache should create a new instance
        with patch("api.oauth2_config.OAUTH2_ISSUER", "https://new-issuer.example.com"), \
             patch("api.oauth2_config.OAUTH2_JWKS_URI", ""):
            from api.oauth2 import _get_jwks_cache
            new_cache = _get_jwks_cache()
            assert new_cache.jwks_uri == "https://new-issuer.example.com/.well-known/jwks.json"
