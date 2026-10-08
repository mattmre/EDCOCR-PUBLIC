"""OAuth2/OIDC configuration parsed from environment variables.

All settings have sensible defaults.  OAuth2 is disabled by default
(``OAUTH2_ENABLED=false``) so existing API-key-only deployments are
unaffected.
"""

from __future__ import annotations

import os

# --- Master toggle ---
OAUTH2_ENABLED: bool = os.environ.get("OAUTH2_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- OIDC issuer ---
# Example: https://login.microsoftonline.com/{tenant}/v2.0
OAUTH2_ISSUER: str = os.environ.get("OAUTH2_ISSUER", "")

# --- Audience (aud claim) ---
OAUTH2_AUDIENCE: str = os.environ.get("OAUTH2_AUDIENCE", "")

_DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
if (
    OAUTH2_ENABLED
    and _DEPLOYMENT_ENV in {"production", "prod"}
    and (not OAUTH2_ISSUER or not OAUTH2_AUDIENCE)
):
    raise RuntimeError(
        "OAUTH2_ISSUER and OAUTH2_AUDIENCE are required when OAuth2 is enabled in production"
    )

# --- JWKS endpoint ---
# If empty, derived from OAUTH2_ISSUER + /.well-known/openid-configuration
OAUTH2_JWKS_URI: str = os.environ.get("OAUTH2_JWKS_URI", "")

# --- Algorithms accepted for JWT signature verification ---
OAUTH2_ALGORITHMS: list[str] = [
    alg.strip()
    for alg in os.environ.get("OAUTH2_ALGORITHMS", "RS256").split(",")
    if alg.strip()
]

# --- Role claim path inside the JWT payload ---
OAUTH2_ROLE_CLAIM: str = os.environ.get("OAUTH2_ROLE_CLAIM", "roles")

# --- Optional tenant claim path (RC-02) ---
# When set, WebSocket/HTTP multi-tenancy may pin request.state.tenant_id from
# this JWT claim after signature validation. Empty = OAuth is not a tenant
# authority (fail-closed under MT without API-key tenant pin).
OAUTH2_TENANT_CLAIM: str = os.environ.get("OAUTH2_TENANT_CLAIM", "").strip()

# --- Role mapping: OIDC claim value -> internal role name ---
OAUTH2_ADMIN_ROLE: str = os.environ.get("OAUTH2_ADMIN_ROLE", "admin")
OAUTH2_OPERATOR_ROLE: str = os.environ.get("OAUTH2_OPERATOR_ROLE", "operator")
OAUTH2_VIEWER_ROLE: str = os.environ.get("OAUTH2_VIEWER_ROLE", "viewer")

# --- Default role for JWT users without a matching role claim ---
OAUTH2_DEFAULT_ROLE: str = os.environ.get("OAUTH2_DEFAULT_ROLE", "viewer")

# --- Role assigned to API key users (backward compatibility) ---
APIKEY_ROLE: str = os.environ.get("APIKEY_ROLE", "operator")
