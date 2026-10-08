"""
Django settings for coordinator project.

Distributed OCR pipeline coordinator service.
"""

import math
import os
import re
import ssl

# S1-A09: single-path credentials — vault/aws before env when configured.
try:
    from credential_manager import apply_runtime_credentials_to_environ

    apply_runtime_credentials_to_environ()
except Exception:
    pass
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from django.core.exceptions import ImproperlyConfigured

# ---------------------------------------------------------------------------
# Credential manager bridge (vault / KMS / env fallback chain)
# ---------------------------------------------------------------------------
try:
    from coordinator.credential_bridge import get_credential as _get_credential
except ImportError:
    # Standalone import fallback (e.g. running outside the coordinator package)
    def _get_credential(key, default=None):  # type: ignore[misc]
        return os.environ.get(key, default)


BASE_DIR = Path(__file__).resolve().parent.parent


def _get_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(f"{name} must be an integer >= 1") from exc
    if value < 1:
        raise ImproperlyConfigured(f"{name} must be >= 1")
    return value


def _get_nonnegative_float_env(
    name: str,
    default: float,
    *,
    maximum: float,
) -> float:
    """Parse a finite nonnegative float and reject impossible capacities."""
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(f"{name} must be a finite number >= 0") from exc
    if not math.isfinite(value) or value < 0 or value > maximum:
        raise ImproperlyConfigured(
            f"{name} must be a finite number between 0 and {maximum:g}"
        )
    return value


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _secure_transport_url(
    value: str,
    *,
    plain_scheme: str,
    secure_scheme: str,
    plain_port: int,
    secure_port: int,
) -> str:
    """Upgrade a legacy plaintext URL when the TLS env contract is enabled."""
    parsed = urlparse(value)
    if parsed.scheme != plain_scheme:
        return value
    host_port = parsed.netloc.rsplit("@", 1)[-1]
    if host_port.startswith("["):
        host_without_port = host_port.split("]", 1)[0] + "]"
    else:
        host_without_port = (
            host_port.rsplit(":", 1)[0] if ":" in host_port else host_port
        )
    current_port = None
    try:
        current_port = parsed.port
    except ValueError:
        return value
    target_port = secure_port if current_port in (None, plain_port) else current_port
    replacement = f"{host_without_port}:{target_port}"
    if "@" in parsed.netloc:
        replacement = parsed.netloc.rsplit("@", 1)[0] + "@" + replacement
    return parsed._replace(scheme=secure_scheme, netloc=replacement).geturl()


# Security

SECRET_KEY = _get_credential("DJANGO_SECRET_KEY", "")
if not SECRET_KEY and os.environ.get("DJANGO_DEBUG", "").lower() not in ("true", "1"):
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY environment variable must be set in production. "
        "Set DJANGO_DEBUG=True for development without a secret key."
    )
if not SECRET_KEY:
    SECRET_KEY = (
        "django-insecure-dev-only-do-not-use-in-production"  # pragma: allowlist secret
    )

DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() in ("true", "1", "yes")

DEPLOYMENT_ENV = os.environ.get("DEPLOYMENT_ENV", "development").strip().lower()
if DEPLOYMENT_ENV not in {"development", "staging", "production"}:
    raise ImproperlyConfigured(
        "DEPLOYMENT_ENV must be one of: development, staging, production."
    )

if DEPLOYMENT_ENV == "production":
    if DEBUG:
        raise ImproperlyConfigured(
            "DJANGO_DEBUG must be False when DEPLOYMENT_ENV=production."
        )
    if os.environ.get("PRODUCTION_READINESS_ACK", "").lower() not in (
        "true",
        "1",
        "yes",
    ):
        raise ImproperlyConfigured(
            "DEPLOYMENT_ENV=production requires PRODUCTION_READINESS_ACK=true. "
            "Run docs/deployment/distributed-readiness-checklist.md first."
        )

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_celery_results",
    "django_celery_beat",
    "django_otp",
    "django_otp.plugins.otp_totp",
    "jobs",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django_otp.middleware.OTPMiddleware",
    "jobs.mfa.MFAMiddleware",
    # R4-RC-07/08 / S2-A08: pin Postgres app.tenant_id for FORCE RLS
    "jobs.rls_middleware.TenantRLSMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "coordinator.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "coordinator.wsgi.application"

# Database — PostgreSQL.  The co-packaged FastAPI process reserves
# DATABASE_URL for its SQLite-only SQLAlchemy store, so it supplies the
# coordinator connection through the explicit COORDINATOR_DATABASE_URL alias.

_database_url = os.environ.get("COORDINATOR_DATABASE_URL", "") or os.environ.get(
    "DATABASE_URL", ""
)
if not _database_url:
    raise ImproperlyConfigured(
        "COORDINATOR_DATABASE_URL or DATABASE_URL must be set. "
        "Example: postgres://DB_USER@DB_HOST:5432/DB_NAME (password via URL userinfo)"
    )
_parsed_db = urlparse(_database_url)
_database_sslmode = os.environ.get("DATABASE_SSLMODE", "").strip()
_database_sslrootcert = os.environ.get("DATABASE_SSLROOTCERT", "").strip()
DATABASE_SCHEMA = os.environ.get("DATABASE_SCHEMA", "public").strip()
if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", DATABASE_SCHEMA):
    raise ImproperlyConfigured(
        "DATABASE_SCHEMA must be a lowercase PostgreSQL identifier containing "
        "only letters, digits, and underscores (maximum 63 characters)."
    )
_database_options = {}
if _database_sslmode:
    _database_options["sslmode"] = _database_sslmode
if _database_sslrootcert:
    _database_options["sslrootcert"] = _database_sslrootcert
_database_options["options"] = f"-c search_path={DATABASE_SCHEMA}"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _parsed_db.path.lstrip("/"),
        "USER": _parsed_db.username or "",
        "PASSWORD": _parsed_db.password or "",
        "HOST": _parsed_db.hostname or "localhost",
        "PORT": str(_parsed_db.port or 5432),
        "OPTIONS": _database_options,
        "CONN_MAX_AGE": 300,  # Reuse DB connections in Celery workers (5 min)
        "CONN_HEALTH_CHECKS": True,  # Django 5.2+: validate stale connections before use
    }
}

# Cache — Redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
_REDIS_SSL_CA = os.environ.get(
    "REDIS_SSL_CA",
    os.environ.get("DATABASE_SSLROOTCERT", "/etc/tls/ca.crt"),
)
if _env_truthy("REDIS_SSL"):
    REDIS_URL = _secure_transport_url(
        REDIS_URL,
        plain_scheme="redis",
        secure_scheme="rediss",
        plain_port=6379,
        secure_port=6380,
    )

CACHES = {
    "default": {
        "BACKEND": (
            "jobs.redis_client.SentinelRedisCache"
            if os.environ.get("REDIS_SENTINEL_MASTER_NAME", "").strip()
            else "django.core.cache.backends.redis.RedisCache"
        ),
        "LOCATION": REDIS_URL,
    }
}
if _env_truthy("REDIS_SSL"):
    CACHES["default"]["OPTIONS"] = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
        "ssl_ca_certs": _REDIS_SSL_CA,
    }

# Password validation

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Internationalization

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Default primary key field type

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Celery

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")
if not CELERY_BROKER_URL:
    raise ImproperlyConfigured(
        "CELERY_BROKER_URL environment variable must be set. "
        "Example: amqp://RABBIT_USER@rabbitmq:5672// (password via URL userinfo)"
    )
if _env_truthy("CELERY_BROKER_USE_SSL"):
    CELERY_BROKER_URL = _secure_transport_url(
        CELERY_BROKER_URL,
        plain_scheme="amqp",
        secure_scheme="amqps",
        plain_port=5672,
        secure_port=5671,
    )
    if urlparse(CELERY_BROKER_URL).scheme == "amqps":
        CELERY_BROKER_USE_SSL = {
            "cert_reqs": ssl.CERT_REQUIRED,
            "ca_certs": os.environ.get("CELERY_BROKER_SSL_CA", _REDIS_SSL_CA),
        }
# RC-05 / R2-RC-01 residual: production refuses bootstrap/admin broker users.
# ImportError is fail-closed under production / REQUIRE_BROKER_LEAST_PRIV;
# lab/dev may soft-skip only when enforcement is off.
_deployment_env = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
_require_broker_lp = os.environ.get(
    "REQUIRE_BROKER_LEAST_PRIV", ""
).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_broker_lp_enforce = _require_broker_lp or _deployment_env in ("production", "prod")
try:
    from jobs.broker_least_priv import (  # noqa: WPS433
        BrokerLeastPrivError,
        assert_broker_url_least_priv,
    )

    assert_broker_url_least_priv(CELERY_BROKER_URL, role="celery")
except ImportError as exc:
    if _broker_lp_enforce:
        raise ImproperlyConfigured(
            "RC-05: jobs.broker_least_priv import failed under broker "
            f"least-priv enforcement: {exc}"
        ) from exc
except BrokerLeastPrivError as exc:
    raise ImproperlyConfigured(str(exc)) from exc
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
if _env_truthy("REDIS_SSL"):
    CELERY_RESULT_BACKEND = _secure_transport_url(
        CELERY_RESULT_BACKEND,
        plain_scheme="redis",
        secure_scheme="rediss",
        plain_port=6379,
        secure_port=6380,
    )
    CELERY_REDIS_BACKEND_USE_SSL = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
        "ssl_ca_certs": _REDIS_SSL_CA,
    }

# Redis Sentinel support (Phase 7D)
# When using Sentinel, set CELERY_RESULT_BACKEND=sentinel://sentinel1:26379/0
# and REDIS_SENTINEL_MASTER_NAME=ocr-master
_REDIS_SENTINEL_MASTER = os.environ.get("REDIS_SENTINEL_MASTER_NAME", "")
if _REDIS_SENTINEL_MASTER:
    _sentinel_password = _get_credential("REDIS_SENTINEL_PASSWORD", "")
    CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = {
        "master_name": _REDIS_SENTINEL_MASTER,
    }
    if _sentinel_password:
        CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS["sentinel_kwargs"] = {
            "password": _sentinel_password,
        }

CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
# R2-RC-01: application-level HMAC task authenticity (see jobs.task_signing).
# JSON serializers stay locked (no pickle). Signing is orthogonal to content-type.
CELERY_TASK_SIGNING_KEY = os.environ.get("CELERY_TASK_SIGNING_KEY", "").strip()
ENABLE_CELERY_TASK_SIGNING = os.environ.get(
    "ENABLE_CELERY_TASK_SIGNING", ""
).lower() in ("1", "true", "yes")

CELERY_USE_QUORUM_QUEUES = os.environ.get(
    "CELERY_USE_QUORUM_QUEUES", "False"
).lower() in ("true", "1", "yes")
MAX_PROCESSING_TIMEOUT_MINUTES = 10_080
JOB_PROCESSING_TIMEOUT_MINUTES = _get_positive_int_env(
    "JOB_PROCESSING_TIMEOUT_MINUTES",
    30,
)
JOB_PROCESSING_TIMEOUT_MINUTES = min(
    JOB_PROCESSING_TIMEOUT_MINUTES,
    MAX_PROCESSING_TIMEOUT_MINUTES,
)
# Regex-only NER is an explicitly visible degraded result. Production remains
# fail-closed unless an operator chooses continuity with custody evidence.
ALLOW_DEGRADED_NER = _env_truthy("ALLOW_DEGRADED_NER")

# Celery Beat schedule
OCR_STAGE_RECOVERY_INTERVAL_SECONDS = max(
    5, min(_get_positive_int_env("OCR_STAGE_RECOVERY_INTERVAL_SECONDS", 60), 300)
)
CELERY_BEAT_SCHEDULE = {
    "check-worker-heartbeats": {
        "task": "jobs.tasks.check_worker_heartbeats",
        "schedule": 60.0,  # Every 60 seconds
    },
    "relay-task-dispatch-outbox": {
        "task": "jobs.tasks.relay_task_dispatch_outbox",
        "schedule": 30.0,  # Repair DB-commit/broker-publish crash windows
    },
    "cleanup-stale-jobs": {
        "task": "jobs.tasks.cleanup_stale_jobs",
        "schedule": 300.0,  # Every 5 minutes
    },
    "recover-page-stages": {
        "task": "jobs.tasks.recover_page_stages",
        "schedule": float(OCR_STAGE_RECOVERY_INTERVAL_SECONDS),
    },
    # MP-B1: redrive orphaned assembling jobs (lost post-assembly chord)
    "redrive-stale-assembling-jobs": {
        "task": "jobs.tasks.redrive_stale_assembling_jobs",
        "schedule": 60.0,  # Every 60 seconds
    },
    "cleanup-completed-jobs": {
        "task": "jobs.tasks.cleanup_completed_jobs",
        "schedule": 86400.0,  # Every 24 hours
    },
    "cleanup-expired-presigned-capabilities": {
        "task": "jobs.tasks.cleanup_expired_presigned_capabilities",
        "schedule": 300.0,  # Every 5 minutes; bearer material expires quickly
    },
    "cleanup-pii-entities": {
        "task": "jobs.tasks.cleanup_pii_entities",
        "schedule": 86400.0,  # Every 24 hours
    },
    "cleanup-output-files": {
        "task": "jobs.tasks.cleanup_output_files",
        "schedule": 604800.0,  # Every 7 days
    },
    "rotate-audit-logs": {
        "task": "jobs.tasks.rotate_audit_logs_task",
        "schedule": 2592000.0,  # Every 30 days
    },
    "republish-pending-custody": {
        "task": "jobs.tasks.republish_pending_custody",
        "schedule": 300.0,  # Every 5 minutes; bounded crash-gap repair
    },
}

# Storage backends (Phase 7 kickoff)
# Backward compatible default is NFS.

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "nfs").strip().lower()
NFS_ROOT = os.environ.get("NFS_ROOT", "/shared")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ACCESS_KEY = _get_credential("S3_ACCESS_KEY", "")
S3_SECRET_KEY = _get_credential("S3_SECRET_KEY", "")
S3_SESSION_TOKEN = _get_credential("S3_SESSION_TOKEN", "")
S3_AUTH_MODE = os.environ.get("S3_AUTH_MODE", "static").strip().lower()
S3_REGION = os.environ.get("S3_REGION", "")

# Worker-local S3 download cache (avoids redundant downloads in fan-out).
# maintenance review: use tempfile.gettempdir() instead of hardcoded /tmp and restrict
# the directory to owner-only (mode 0o700) to prevent cache poisoning on
# shared hosts.  Override path via S3_CACHE_DIR env var.
S3_CACHE_DIR = os.environ.get(
    "S3_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "ocr-s3-cache"),
)
try:
    os.makedirs(S3_CACHE_DIR, mode=0o700, exist_ok=True)
    # Enforce restrictive permissions even if directory pre-existed.
    try:
        os.chmod(S3_CACHE_DIR, 0o700)
    except (OSError, NotImplementedError):
        # chmod is a no-op on Windows; ignore.
        pass
except OSError:
    # Directory creation is best-effort at import time; storage backend
    # code will surface a clearer error at actual use.
    pass
S3_CACHE_MAX_SIZE_GB = _get_nonnegative_float_env(
    "S3_CACHE_MAX_SIZE_GB", 1.0, maximum=10_000.0
)
S3_CACHE_MAX_OBJECT_GB = _get_nonnegative_float_env(
    "S3_CACHE_MAX_OBJECT_GB", 0.5, maximum=10_000.0
)

# Presigned URL mode for credential-free workers (Phase 7C)
S3_USE_PRESIGNED_URLS = os.environ.get("S3_USE_PRESIGNED_URLS", "false").lower() in (
    "true",
    "1",
    "yes",
)
S3_PRESIGNED_ONLY_WORKER = os.environ.get("S3_PRESIGNED_ONLY_WORKER", "false").lower() in (
    "true", "1", "yes",
)
_raw_expiry = _get_positive_int_env("S3_PRESIGNED_URL_EXPIRY", 300)
S3_PRESIGNED_URL_EXPIRY = max(60, min(_raw_expiry, 86400))
# Opaque job/operation capabilities are revocable database references, not
# bearer URLs.  Keep them alive across queue residence and signed-task
# redelivery; each actual URL is issued just in time with the shorter TTL.
_raw_capability_expiry = _get_positive_int_env("S3_PRESIGNED_CAPABILITY_EXPIRY", 86400)
S3_PRESIGNED_CAPABILITY_EXPIRY = max(
    S3_PRESIGNED_URL_EXPIRY,
    min(_raw_capability_expiry, 604800),
)
S3_PRESIGNED_TASK_KEY = os.environ.get("S3_PRESIGNED_TASK_KEY", "").strip()

if STORAGE_BACKEND not in ("nfs", "s3"):
    raise ImproperlyConfigured("STORAGE_BACKEND must be one of: 'nfs', 's3'")

if S3_PRESIGNED_ONLY_WORKER:
    if STORAGE_BACKEND != "s3" or not S3_USE_PRESIGNED_URLS:
        raise ImproperlyConfigured("S3_PRESIGNED_ONLY_WORKER requires S3 presigned URL mode")
    if os.environ.get("AWS_EC2_METADATA_DISABLED", "").lower() != "true":
        raise ImproperlyConfigured("presigned-only worker requires AWS_EC2_METADATA_DISABLED=true")
    if S3_ACCESS_KEY or S3_SECRET_KEY or S3_SESSION_TOKEN or any(
        os.environ.get(name) for name in (
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
            "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        )
    ):
        raise ImproperlyConfigured("presigned-only worker must not have S3 or ambient AWS credentials")

if STORAGE_BACKEND == "s3":
    if S3_AUTH_MODE not in {"static", "iam"}:
        raise ImproperlyConfigured("S3_AUTH_MODE must be static or iam")
    if S3_AUTH_MODE == "iam" and (S3_ACCESS_KEY or S3_SECRET_KEY or S3_SESSION_TOKEN):
        raise ImproperlyConfigured("S3_AUTH_MODE=iam must not receive explicit S3 credentials")
    # IAM uses the SDK's refreshable provider chain. Empty endpoint means
    # the regional AWS endpoint, rather than a required MinIO override.
    needs_static_keys = S3_AUTH_MODE == "static" and not S3_PRESIGNED_ONLY_WORKER
    missing = [
        name
        for name, value in (
            ("S3_ENDPOINT", S3_ENDPOINT if S3_AUTH_MODE == "static" else "aws-default"),
            ("S3_BUCKET", S3_BUCKET),
            ("S3_ACCESS_KEY", S3_ACCESS_KEY if needs_static_keys else "provider-or-presigned"),
            ("S3_SECRET_KEY", S3_SECRET_KEY if needs_static_keys else "provider-or-presigned"),
        )
        if not value
    ]
    if missing:
        raise ImproperlyConfigured(
            "STORAGE_BACKEND=s3 requires env vars: " + ", ".join(missing)
        )

if S3_USE_PRESIGNED_URLS and STORAGE_BACKEND != "s3":
    raise ImproperlyConfigured("S3_USE_PRESIGNED_URLS=true requires STORAGE_BACKEND=s3")
if S3_USE_PRESIGNED_URLS:
    if S3_AUTH_MODE == "iam" and not os.environ.get("S3_PRESIGNED_ALLOWED_HOSTS", "").strip():
        raise ImproperlyConfigured("IAM presigned mode requires explicit S3_PRESIGNED_ALLOWED_HOSTS")
    if not S3_PRESIGNED_TASK_KEY:
        raise ImproperlyConfigured(
            "S3_USE_PRESIGNED_URLS=true requires S3_PRESIGNED_TASK_KEY"
        )
    try:
        from cryptography.fernet import Fernet

        Fernet(S3_PRESIGNED_TASK_KEY.encode("ascii"))
    except (ImportError, ValueError, UnicodeEncodeError) as exc:
        raise ImproperlyConfigured(
            "S3_PRESIGNED_TASK_KEY must be a valid Fernet key"
        ) from exc

# Context windowing (Phase 2: 5-page context windows for cross-page merging)
CONTEXT_WINDOW_ENABLED = os.environ.get("CONTEXT_WINDOW_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
CONTEXT_WINDOW_SIZE = _get_positive_int_env("CONTEXT_WINDOW_SIZE", 5)
CONTEXT_STORE_TTL = _get_positive_int_env("CONTEXT_STORE_TTL", 3600)
CONTEXT_STORE_URL = os.environ.get("CONTEXT_STORE_URL", "") or REDIS_URL
