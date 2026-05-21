"""API configuration with environment variable overrides."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _first_env(*keys: str, default: str) -> str:
    """Return the first non-empty environment value among the given keys."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return default


# Canonical env-parsing helpers (DRY consolidation)
from ocr_distributed.ocr_utils import get_env_float as _safe_float
from ocr_distributed.ocr_utils import get_env_int as _safe_int


def _parse_csv_values(raw: str) -> tuple[str, ...]:
    """Parse comma-separated config values into a normalized tuple."""
    values = [item.strip() for item in raw.split(",")]
    return tuple(item for item in values if item)


# --- Server ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = _safe_int("API_PORT", 8000, min_val=1, max_val=65535)

# --- Paths ---
SOURCE_FOLDER = _first_env("SOURCE_FOLDER", "OCR_SOURCE_DIR", default="/app/ocr_source")
OUTPUT_FOLDER = _first_env("OUTPUT_FOLDER", "OCR_OUTPUT_DIR", default="/app/ocr_output")
DB_PATH = _first_env("API_DB_PATH", default=os.path.join(OUTPUT_FOLDER, "jobs.db"))

# --- Limits ---
MAX_UPLOAD_SIZE_MB = _safe_int("MAX_UPLOAD_SIZE_MB", 5120, max_val=51200)
MAX_CONCURRENT_JOBS = _safe_int("MAX_CONCURRENT_JOBS", 4, max_val=64)
MAX_BATCH_SIZE = _safe_int("MAX_BATCH_SIZE", 50, max_val=500)

# DoS-hardening cap on non-multipart request bodies (bytes).
# Multipart uploads are exempt and governed by MAX_UPLOAD_SIZE_MB instead.
# Default: 10 MiB.  Set to 0 to disable the check.
MAX_REQUEST_BODY_SIZE = _safe_int(
    "MAX_REQUEST_BODY_SIZE",
    10 * 1024 * 1024,
    min_val=0,
    max_val=10 * 1024 * 1024 * 1024,  # 10 GiB upper bound
)

# --- API Security ---
OCR_API_KEY = os.environ.get("OCR_API_KEY", "").strip()
ALLOW_UNAUTHENTICATED = os.environ.get("ALLOW_UNAUTHENTICATED", "").lower() in (
    "1",
    "true",
    "yes",
)
ANONYMOUS_ROLE = os.environ.get("ANONYMOUS_ROLE", "viewer").strip() or "viewer"
API_ALLOWED_IPS = _parse_csv_values(os.environ.get("API_ALLOWED_IPS", ""))
CSP_POLICY = os.environ.get("CSP_POLICY", "default-src 'self'").strip() or "default-src 'self'"

# --- OpenAPI docs exposure ---
# By default the interactive docs (/docs, /redoc) and the raw schema
# (/openapi.json) are NOT mounted, so that an unauthenticated caller cannot
# enumerate the full API surface.  Set EXPOSE_API_DOCS=true to restore the
# legacy behaviour (useful for internal dev environments with trusted
# network ingress).
EXPOSE_API_DOCS = os.environ.get("EXPOSE_API_DOCS", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- CORS ---
# Comma-separated list of allowed origins (e.g., "https://app.example.com,https://admin.example.com").
# Empty string (default) means CORS middleware is not added -- no cross-origin access.
CORS_ALLOWED_ORIGINS: tuple[str, ...] = _parse_csv_values(
    os.environ.get("CORS_ALLOWED_ORIGINS", "")
)

API_AUDIT_LOG_ENABLED = os.environ.get("API_AUDIT_LOG_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
)
API_AUDIT_LOG_PATH = os.environ.get("API_AUDIT_LOG_PATH", "").strip()
API_AUDIT_EXCLUDE_HEALTH = os.environ.get("API_AUDIT_EXCLUDE_HEALTH", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- Pipeline ---
PIPELINE_SCRIPT = os.environ.get(
    "PIPELINE_SCRIPT",
    str(Path(__file__).resolve().parent.parent / "ocr_gpu_async.py"),
)
PIPELINE_POLL_INTERVAL = _safe_int("PIPELINE_POLL_INTERVAL", 5, max_val=300)
JOB_PROCESSING_TIMEOUT_MINUTES = _safe_int(
    "JOB_PROCESSING_TIMEOUT_MINUTES",
    30,
    max_val=10_080,
)

# --- Cleanup ---
RESULT_RETENTION_DAYS = _safe_int("RESULT_RETENTION_DAYS", 90, max_val=3650)

# --- Webhooks ---
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip()
WEBHOOK_TIMEOUT = _safe_int("WEBHOOK_TIMEOUT", 30, max_val=120)
WEBHOOK_MAX_RETRIES = _safe_int("WEBHOOK_MAX_RETRIES", 3, max_val=10)
WEBHOOK_ALLOW_HTTP = os.environ.get("WEBHOOK_ALLOW_HTTP", "").lower() in (
    "1",
    "true",
    "yes",
)
WEBHOOK_ALLOW_PRIVATE = os.environ.get("WEBHOOK_ALLOW_PRIVATE", "").lower() in (
    "1",
    "true",
    "yes",
)
WEBHOOK_ENRICH_ENTITIES = os.environ.get("WEBHOOK_ENRICH_ENTITIES", "").lower() in (
    "1",
    "true",
    "yes",
)
WEBHOOK_SECRET_KEY = os.environ.get("WEBHOOK_SECRET_KEY", "").strip()


# --- Webhook secret encryption helpers ---


def _get_webhook_encryption_key() -> bytes:
    """Derive a Fernet-compatible encryption key for webhook secrets.

    Uses ``WEBHOOK_SECRET_KEY`` env var if set, otherwise falls back to
    ``OCR_API_KEY``.  A deterministic 32-byte key is derived via SHA-256
    and then base64-encoded for Fernet.
    """
    raw = WEBHOOK_SECRET_KEY or OCR_API_KEY or "dev-key-not-for-production"
    key_bytes = hashlib.sha256(raw.encode()).digest()
    return base64.urlsafe_b64encode(key_bytes)


def encrypt_webhook_secret(plaintext: str) -> str:
    """Encrypt a webhook secret for at-rest storage.

    Returns a Fernet token string (URL-safe base64).
    """
    from cryptography.fernet import Fernet

    fernet = Fernet(_get_webhook_encryption_key())
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_webhook_secret(ciphertext: str) -> str:
    """Decrypt a stored webhook secret.

    Falls back to returning the value as-is when decryption fails,
    which handles pre-migration plaintext secrets.  A warning is logged
    to prompt secret rotation.
    """
    from cryptography.fernet import Fernet

    fernet = Fernet(_get_webhook_encryption_key())
    try:
        return fernet.decrypt(ciphertext.encode()).decode()
    except Exception:
        logger.warning(
            "Webhook secret decryption failed -- treating as plaintext "
            "(pre-migration value). Rotate secrets to enable encryption."
        )
        return ciphertext


# --- Durable Event Stream ---
API_EVENT_STREAM_ENABLED = os.environ.get("API_EVENT_STREAM_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)
API_EVENT_STREAM_PATH = os.environ.get(
    "API_EVENT_STREAM_PATH",
    os.path.join(OUTPUT_FOLDER, "logs", "api-events.jsonl"),
)

# --- Durable Event Store (SQLite-backed for replay) ---
EVENT_STORE_ENABLED = os.environ.get("EVENT_STORE_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
)
EVENT_STORE_PATH = os.environ.get(
    "EVENT_STORE_PATH",
    os.path.join(OUTPUT_FOLDER, "event_store.db"),
)
EVENT_RETENTION_HOURS = _safe_int("EVENT_RETENTION_HOURS", 72, max_val=8760)

# --- Webhook Dead-Letter Queue ---
WEBHOOK_DLQ_ENABLED = os.environ.get("WEBHOOK_DLQ_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
)
WEBHOOK_DLQ_PATH = os.environ.get(
    "WEBHOOK_DLQ_PATH",
    os.path.join(OUTPUT_FOLDER, "logs", "webhook_dlq.jsonl"),
)

# --- OAuth2 / OIDC ---
OAUTH2_ENABLED = os.environ.get("OAUTH2_ENABLED", "").lower() in (
    "1",
    "true",
    "yes",
)

# --- Feature Flags ---
# Transform and stamping support (Phase A+)
ENABLE_TRANSFORMS = os.environ.get("ENABLE_TRANSFORMS", "").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_STAMPING = os.environ.get("ENABLE_STAMPING", "").lower() in (
    "1",
    "true",
    "yes",
)

# Multi-tenancy support (opt-in)
ENABLE_MULTITENANCY = os.environ.get("ENABLE_MULTITENANCY", "").lower() in (
    "1",
    "true",
    "yes",
)

# Internal tenant-cost accounting (provider-agnostic; safe zero defaults)
TENANT_COST_PER_PAGE_USD = _safe_float("TENANT_COST_PER_PAGE_USD", 0.0)
TENANT_COST_PER_GIB_INGESTED_USD = _safe_float(
    "TENANT_COST_PER_GIB_INGESTED_USD",
    0.0,
)
TENANT_COST_PER_API_CALL_USD = _safe_float("TENANT_COST_PER_API_CALL_USD", 0.0)
TENANT_COST_PER_PROCESSING_HOUR_USD = _safe_float(
    "TENANT_COST_PER_PROCESSING_HOUR_USD",
    0.0,
)

# Tenant-facing SLO monitoring
TENANT_SLO_WINDOW_HOURS = _safe_int(
    "TENANT_SLO_WINDOW_HOURS",
    24,
    min_val=1,
    max_val=24 * 30,
)
TENANT_SLO_TARGET_SUCCESS_RATE = _safe_float(
    "TENANT_SLO_TARGET_SUCCESS_RATE",
    0.95,
    min_val=0.0,
    max_val=1.0,
)
TENANT_SLO_TARGET_P95_PROCESSING_SECONDS = _safe_float(
    "TENANT_SLO_TARGET_P95_PROCESSING_SECONDS",
    1800.0,
    min_val=1.0,
)
