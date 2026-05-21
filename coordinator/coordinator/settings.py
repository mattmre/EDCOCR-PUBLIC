"""
Django settings for coordinator project.

Distributed OCR pipeline coordinator service.
"""

import os
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

# Security

SECRET_KEY = _get_credential('DJANGO_SECRET_KEY', '')
if not SECRET_KEY and os.environ.get('DJANGO_DEBUG', '').lower() not in ('true', '1'):
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY environment variable must be set in production. "
        "Set DJANGO_DEBUG=True for development without a secret key."
    )
if not SECRET_KEY:
    SECRET_KEY = 'django-insecure-dev-only-do-not-use-in-production'

DEBUG = os.environ.get('DJANGO_DEBUG', 'False').lower() in ('true', '1', 'yes')

DEPLOYMENT_ENV = os.environ.get('DEPLOYMENT_ENV', 'development').strip().lower()
if DEPLOYMENT_ENV not in {'development', 'staging', 'production'}:
    raise ImproperlyConfigured(
        "DEPLOYMENT_ENV must be one of: development, staging, production."
    )

if DEPLOYMENT_ENV == 'production':
    if DEBUG:
        raise ImproperlyConfigured(
            "DJANGO_DEBUG must be False when DEPLOYMENT_ENV=production."
        )
    if os.environ.get('PRODUCTION_READINESS_ACK', '').lower() not in ('true', '1', 'yes'):
        raise ImproperlyConfigured(
            "DEPLOYMENT_ENV=production requires PRODUCTION_READINESS_ACK=true. "
            "Run docs/deployment/distributed-readiness-checklist.md first."
        )

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    if h.strip()
]

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'django_celery_beat',
    'django_otp',
    'django_otp.plugins.otp_totp',
    'jobs',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django_otp.middleware.OTPMiddleware',
    'jobs.mfa.MFAMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'coordinator.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'coordinator.wsgi.application'

# Database — PostgreSQL from DATABASE_URL

_database_url = os.environ.get('DATABASE_URL', '')
if not _database_url:
    raise ImproperlyConfigured(
        "DATABASE_URL environment variable must be set. "
        "Example: postgres://user:password@host:5432/dbname"
    )
_parsed_db = urlparse(_database_url)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': _parsed_db.path.lstrip('/'),
        'USER': _parsed_db.username or '',
        'PASSWORD': _parsed_db.password or '',
        'HOST': _parsed_db.hostname or 'localhost',
        'PORT': str(_parsed_db.port or 5432),
        'CONN_MAX_AGE': 300,  # Reuse DB connections in Celery workers (5 min)
        'CONN_HEALTH_CHECKS': True,  # Django 5.2+: validate stale connections before use
    }
}

# Cache — Redis

REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': REDIS_URL,
    }
}

# Password validation

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Celery

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', '')
if not CELERY_BROKER_URL:
    raise ImproperlyConfigured(
        "CELERY_BROKER_URL environment variable must be set. "
        "Example: amqp://user:password@rabbitmq:5672//"
    )
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# Redis Sentinel support (Phase 7D)
# When using Sentinel, set CELERY_RESULT_BACKEND=sentinel://sentinel1:26379/0
# and REDIS_SENTINEL_MASTER_NAME=ocr-master
_REDIS_SENTINEL_MASTER = os.environ.get('REDIS_SENTINEL_MASTER_NAME', '')
if _REDIS_SENTINEL_MASTER:
    _sentinel_password = _get_credential('REDIS_SENTINEL_PASSWORD', '')
    CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS = {
        'master_name': _REDIS_SENTINEL_MASTER,
    }
    if _sentinel_password:
        CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS['sentinel_kwargs'] = {
            'password': _sentinel_password,
        }

CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
CELERY_USE_QUORUM_QUEUES = os.environ.get('CELERY_USE_QUORUM_QUEUES', 'False').lower() in (
    'true', '1', 'yes'
)
JOB_PROCESSING_TIMEOUT_MINUTES = _get_positive_int_env(
    'JOB_PROCESSING_TIMEOUT_MINUTES',
    30,
)

# Celery Beat schedule
CELERY_BEAT_SCHEDULE = {
    'check-worker-heartbeats': {
        'task': 'jobs.tasks.check_worker_heartbeats',
        'schedule': 60.0,  # Every 60 seconds
    },
    'cleanup-stale-jobs': {
        'task': 'jobs.tasks.cleanup_stale_jobs',
        'schedule': 300.0,  # Every 5 minutes
    },
    'cleanup-completed-jobs': {
        'task': 'jobs.tasks.cleanup_completed_jobs',
        'schedule': 86400.0,  # Every 24 hours
    },
    'cleanup-pii-entities': {
        'task': 'jobs.tasks.cleanup_pii_entities',
        'schedule': 86400.0,  # Every 24 hours
    },
    'cleanup-output-files': {
        'task': 'jobs.tasks.cleanup_output_files',
        'schedule': 604800.0,  # Every 7 days
    },
    'rotate-audit-logs': {
        'task': 'jobs.tasks.rotate_audit_logs_task',
        'schedule': 2592000.0,  # Every 30 days
    },
}

# Storage backends (Phase 7 kickoff)
# Backward compatible default is NFS.

STORAGE_BACKEND = os.environ.get('STORAGE_BACKEND', 'nfs').strip().lower()
NFS_ROOT = os.environ.get('NFS_ROOT', '/shared')

S3_ENDPOINT = os.environ.get('S3_ENDPOINT', '')
S3_BUCKET = os.environ.get('S3_BUCKET', '')
S3_ACCESS_KEY = _get_credential('S3_ACCESS_KEY', '')
S3_SECRET_KEY = _get_credential('S3_SECRET_KEY', '')
S3_REGION = os.environ.get('S3_REGION', '')

# Worker-local S3 download cache (avoids redundant downloads in fan-out).
# use tempfile.gettempdir() instead of hardcoded /tmp and restrict
# the directory to owner-only (mode 0o700) to prevent cache poisoning on
# shared hosts.  Override path via S3_CACHE_DIR env var.
S3_CACHE_DIR = os.environ.get(
    'S3_CACHE_DIR',
    os.path.join(tempfile.gettempdir(), 'ocr-s3-cache'),
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
S3_CACHE_MAX_SIZE_GB = int(os.environ.get('S3_CACHE_MAX_SIZE_GB', '10'))

# Presigned URL mode for credential-free workers (Phase 7C)
S3_USE_PRESIGNED_URLS = os.environ.get('S3_USE_PRESIGNED_URLS', 'false').lower() in (
    'true', '1', 'yes'
)
_raw_expiry = int(os.environ.get('S3_PRESIGNED_URL_EXPIRY', '300'))
S3_PRESIGNED_URL_EXPIRY = max(60, min(_raw_expiry, 86400))

if STORAGE_BACKEND not in ('nfs', 's3'):
    raise ImproperlyConfigured(
        "STORAGE_BACKEND must be one of: 'nfs', 's3'"
    )

if STORAGE_BACKEND == 's3':
    missing = [
        name
        for name, value in (
            ('S3_ENDPOINT', S3_ENDPOINT),
            ('S3_BUCKET', S3_BUCKET),
            ('S3_ACCESS_KEY', S3_ACCESS_KEY),
            ('S3_SECRET_KEY', S3_SECRET_KEY),
        )
        if not value
    ]
    if missing:
        raise ImproperlyConfigured(
            "STORAGE_BACKEND=s3 requires env vars: " + ", ".join(missing)
        )

if S3_USE_PRESIGNED_URLS and STORAGE_BACKEND != 's3':
    raise ImproperlyConfigured(
        "S3_USE_PRESIGNED_URLS=true requires STORAGE_BACKEND=s3"
    )

# Context windowing (Phase 2: 5-page context windows for cross-page merging)
CONTEXT_WINDOW_ENABLED = os.environ.get(
    'CONTEXT_WINDOW_ENABLED', 'false'
).lower() in ('1', 'true', 'yes')
CONTEXT_WINDOW_SIZE = int(os.environ.get('CONTEXT_WINDOW_SIZE', '5'))
CONTEXT_STORE_TTL = int(os.environ.get('CONTEXT_STORE_TTL', '3600'))
CONTEXT_STORE_URL = os.environ.get('CONTEXT_STORE_URL', '') or REDIS_URL
