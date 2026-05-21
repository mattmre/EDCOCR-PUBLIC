"""Production credential management with pluggable backends.

Provides a unified interface for retrieving secrets from multiple sources:
- Environment variables (default, always available)
- HashiCorp Vault (optional, requires ``hvac``)
- AWS Secrets Manager / KMS (optional, requires ``boto3``)
- Encrypted credential files (for air-gapped deployments)

Usage::

    from credential_manager import get_credential, validate_credentials

    # Retrieve a single credential (tries backends in priority order)
    db_password = get_credential("POSTGRES_PASSWORD")

    # Validate all production credentials are non-placeholder
    report = validate_credentials()
    if not report.passed:
        for error in report.errors:
            print(error)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CredentialError(Exception):
    """Raised when a credential operation fails in a non-recoverable way.

    Subclassed for specific backend errors (e.g. KMS decryption failures).
    """


# ---------------------------------------------------------------------------
# Credential schema: canonical names for all production secrets
# ---------------------------------------------------------------------------

CREDENTIAL_SCHEMA: dict[str, dict[str, Any]] = {
    # S3 / MinIO object storage
    "S3_ACCESS_KEY": {
        "description": "S3-compatible object storage access key",
        "required_when": "STORAGE_BACKEND=s3",
    },
    "S3_SECRET_KEY": {
        "description": "S3-compatible object storage secret key",
        "required_when": "STORAGE_BACKEND=s3",
    },
    "S3_ENDPOINT": {
        "description": "S3-compatible endpoint URL",
        "required_when": "STORAGE_BACKEND=s3",
    },
    # Django
    "DJANGO_SECRET_KEY": {
        "description": "Django cryptographic signing key",
        "required_when": "always",
    },
    # PostgreSQL
    "POSTGRES_PASSWORD": {
        "description": "PostgreSQL database password",
        "required_when": "always",
    },
    "DATABASE_URL": {
        "description": "PostgreSQL connection URL",
        "required_when": "always",
    },
    # RabbitMQ
    "RABBITMQ_PASSWORD": {
        "description": "RabbitMQ broker password",
        "required_when": "always",
    },
    "CELERY_BROKER_URL": {
        "description": "Celery broker connection URL",
        "required_when": "always",
    },
    # Redis
    "REDIS_PASSWORD": {
        "description": "Redis cache/result backend password",
        "required_when": "optional",
    },
    # API keys
    "METRICS_API_KEY": {
        "description": "Prometheus/metrics endpoint authentication key",
        "required_when": "optional",
    },
    "OCR_API_KEY": {
        "description": "REST API authentication key",
        "required_when": "optional",
    },
}

# Values that should never appear in production credentials
PLACEHOLDER_VALUES: set[str] = {
    "minioadmin",
    "password",
    "secret",
    "changeme",
    "admin",
    "test",
    "default",
}

PLACEHOLDER_SUBSTRINGS: list[str] = [
    "change-me",
    "change_me",
    "changeme",
    "example",
    "your_",
    "placeholder",
    "replace-me",
    "replace_me",
    "xxx",
    "todo",
    "fixme",
]


# ---------------------------------------------------------------------------
# Backend interface and implementations
# ---------------------------------------------------------------------------


class CredentialBackend:
    """Base class for credential retrieval backends."""

    name: str = "base"

    def get(self, key: str) -> Optional[str]:
        """Retrieve a credential value by name. Returns None if not found."""
        raise NotImplementedError

    def available(self) -> bool:
        """Return True if this backend is configured and reachable."""
        return False


class EnvVarBackend(CredentialBackend):
    """Reads credentials from environment variables (always available)."""

    name = "env"

    def get(self, key: str) -> Optional[str]:
        value = os.environ.get(key)
        if value is not None:
            value = value.strip()
            return value if value else None
        return None

    def available(self) -> bool:
        return True


class VaultBackend(CredentialBackend):
    """Reads credentials from HashiCorp Vault (requires ``hvac``).

    Configuration via environment variables:
    - ``VAULT_ADDR``: Vault server URL (e.g. https://vault.example.com:8200)
    - ``VAULT_TOKEN``: Authentication token
    - ``VAULT_SECRET_PATH``: KV secret path (default: ``secret/data/ocr-local``)
    - ``VAULT_NAMESPACE``: Optional Vault namespace for enterprise deployments
    - ``VAULT_KV_VERSION``: KV engine version, ``1`` or ``2`` (default: ``2``)

    For KV v2, the path ``secret/data/<path>`` is used and the response is
    unwrapped from ``.data.data``.  For KV v1, the path ``secret/<path>`` is
    used directly and the response is read from ``.data``.

    Args:
        vault_kv_version: Override the KV engine version (1 or 2).
            When *None*, falls back to ``VAULT_KV_VERSION`` env var
            (default ``2``).
    """

    name = "vault"

    def __init__(self, vault_kv_version: Optional[int] = None) -> None:
        self._client: Any = None
        self._cache: dict[str, str] = {}
        self._loaded = False
        if vault_kv_version is not None:
            self._kv_version = int(vault_kv_version)
        else:
            self._kv_version = int(os.environ.get("VAULT_KV_VERSION", "2"))

    @property
    def vault_kv_version(self) -> int:
        """Return the configured KV engine version (1 or 2)."""
        return self._kv_version

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import hvac  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("hvac not installed; Vault backend unavailable")
            return None
        addr = os.environ.get("VAULT_ADDR", "")
        token = os.environ.get("VAULT_TOKEN", "")
        if not addr or not token:
            logger.debug("VAULT_ADDR or VAULT_TOKEN not set; Vault backend unavailable")
            return None
        namespace = os.environ.get("VAULT_NAMESPACE", "").strip() or None
        try:
            self._client = hvac.Client(url=addr, token=token, namespace=namespace)
            return self._client
        except Exception:
            logger.warning("Failed to create Vault client", exc_info=True)
            return None

    def _load_secrets(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        client = self._get_client()
        if client is None:
            return
        path = os.environ.get("VAULT_SECRET_PATH", "secret/data/ocr-local")
        try:
            if self._kv_version == 1:
                self._load_kv_v1(client, path)
            else:
                self._load_kv_v2(client, path)
        except Exception:
            logger.warning("Failed to read secrets from Vault path %s", path, exc_info=True)

    def _load_kv_v1(self, client: Any, path: str) -> None:
        """Load secrets from a KV v1 engine."""
        # Strip any v2-style ``/data/`` segment that may have been included
        # in the path by convention (users may copy from a v2 example).
        clean = path.replace("/data/", "/", 1) if "/data/" in path else path
        mount = clean.split("/")[0] if "/" in clean else "secret"
        secret_path = "/".join(clean.split("/")[1:]) if "/" in clean else clean
        response = client.secrets.kv.v1.read_secret(
            path=secret_path,
            mount_point=mount,
        )
        data = response.get("data", {})
        if isinstance(data, dict):
            self._cache = {k: str(v) for k, v in data.items()}
            logger.info(
                "Loaded %d secrets from Vault KV v1 path %s", len(self._cache), path
            )

    def _load_kv_v2(self, client: Any, path: str) -> None:
        """Load secrets from a KV v2 engine."""
        # Strip the /data/ segment if present (KV v2 path format)
        clean = path.replace("/data/", "/", 1) if "/data/" in path else path
        # Split into mount point and secret path
        parts = clean.split("/", 1)
        if len(parts) == 2:
            mount, secret_path = parts
        else:
            mount = "secret"
            secret_path = clean
        response = client.secrets.kv.v2.read_secret_version(
            path=secret_path,
            mount_point=mount,
            raise_on_deleted_version=True,
        )
        data = response.get("data", {}).get("data", {})
        if isinstance(data, dict):
            self._cache = {k: str(v) for k, v in data.items()}
            logger.info(
                "Loaded %d secrets from Vault KV v2 path %s", len(self._cache), path
            )

    def get(self, key: str) -> Optional[str]:
        self._load_secrets()
        return self._cache.get(key)

    def available(self) -> bool:
        try:
            import hvac  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            return False
        addr = os.environ.get("VAULT_ADDR", "")
        token = os.environ.get("VAULT_TOKEN", "")
        return bool(addr and token)


class AWSSecretsBackend(CredentialBackend):
    """Reads credentials from AWS Secrets Manager (requires ``boto3``).

    Configuration via environment variables:
    - ``AWS_SECRET_NAME``: Secret name in Secrets Manager
      (default: ``ocr-local/credentials``)
    - ``AWS_REGION_NAME``: AWS region (default: ``us-east-1``)
    - Standard AWS credential chain (``AWS_ACCESS_KEY_ID``, instance role, etc.)
    """

    name = "aws-secrets"

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._loaded = False

    def _load_secrets(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("boto3 not installed; AWS Secrets Manager backend unavailable")
            return
        secret_name = os.environ.get("AWS_SECRET_NAME", "ocr-local/credentials")
        region = os.environ.get("AWS_REGION_NAME", "us-east-1")
        try:
            client = boto3.client("secretsmanager", region_name=region)
            response = client.get_secret_value(SecretId=secret_name)
            secret_string = response.get("SecretString", "")
            if secret_string:
                data = json.loads(secret_string)
                if isinstance(data, dict):
                    self._cache = {k: str(v) for k, v in data.items()}
                    logger.info(
                        "Loaded %d secrets from AWS Secrets Manager (%s)",
                        len(self._cache),
                        secret_name,
                    )
        except Exception:
            logger.warning(
                "Failed to read from AWS Secrets Manager (%s)", secret_name, exc_info=True
            )

    def get(self, key: str) -> Optional[str]:
        self._load_secrets()
        return self._cache.get(key)

    def available(self) -> bool:
        try:
            import boto3  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("AWS_SECRET_NAME", ""))


class AWSKMSBackend(CredentialBackend):
    """Decrypts credential values stored as base64-encoded KMS ciphertexts.

    Each credential is stored as a KMS-encrypted, base64-encoded ciphertext
    in an environment variable prefixed with ``KMS_ENC_`` (e.g.,
    ``KMS_ENC_POSTGRES_PASSWORD``).  The backend strips the prefix, decrypts
    via KMS, and returns the plaintext.

    Configuration via environment variables:
    - ``KMS_ENC_<NAME>``: Base64-encoded KMS ciphertext for credential *NAME*
    - ``AWS_KMS_REGION``: AWS region for the KMS client (default: ``us-east-1``)

    Transient KMS errors (``KMSException`` base class) are retried with
    exponential backoff up to 3 attempts.  Non-transient errors raise
    :class:`CredentialError` with an actionable message.
    """

    name = "aws-kms"
    KMS_PREFIX = "KMS_ENC_"
    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._loaded = False

    def _load_secrets(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.exceptions import ClientError  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("boto3 not installed; AWS KMS backend unavailable")
            return

        region = os.environ.get("AWS_KMS_REGION", "us-east-1")
        kms_client = boto3.client("kms", region_name=region)

        for env_key, env_val in os.environ.items():
            if not env_key.startswith(self.KMS_PREFIX):
                continue
            cred_name = env_key[len(self.KMS_PREFIX) :]
            if not cred_name:
                continue
            try:
                plaintext = self._decrypt_with_retry(
                    kms_client, env_val, cred_name, ClientError
                )
                if plaintext is not None:
                    self._cache[cred_name] = plaintext
            except CredentialError:
                raise
            except Exception:
                logger.warning(
                    "KMS decryption failed for %s", cred_name, exc_info=True
                )

    def _decrypt_with_retry(
        self,
        kms_client: Any,
        ciphertext_b64: str,
        cred_name: str,
        client_error_cls: type,
    ) -> Optional[str]:
        """Decrypt *ciphertext_b64* with exponential backoff retry.

        Raises:
            CredentialError: On non-transient KMS errors
                (``InvalidCiphertextException``, ``AccessDeniedException``).
        """
        ciphertext_blob = base64.b64decode(ciphertext_b64)
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = kms_client.decrypt(CiphertextBlob=ciphertext_blob)
                plaintext_bytes = response["Plaintext"]
                if isinstance(plaintext_bytes, bytes):
                    return plaintext_bytes.decode("utf-8")
                return str(plaintext_bytes)
            except client_error_cls as exc:
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code == "InvalidCiphertextException":
                    raise CredentialError(
                        f"KMS decryption failed for '{cred_name}': ciphertext is "
                        f"corrupt or was encrypted with a different KMS key"
                    ) from exc
                if error_code == "AccessDeniedException":
                    raise CredentialError(
                        f"KMS decryption failed for '{cred_name}': IAM permissions "
                        f"are missing — ensure the service role has kms:Decrypt on "
                        f"the target key"
                    ) from exc
                # Transient KMS errors — retry with backoff
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    backoff = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        "KMS transient error for %s (attempt %d/%d), "
                        "retrying in %ds: %s",
                        cred_name,
                        attempt + 1,
                        self.MAX_RETRIES,
                        backoff,
                        error_code,
                    )
                    time.sleep(backoff)
            except Exception as exc:
                last_exc = exc
                break
        if last_exc is not None:
            logger.warning(
                "KMS decryption exhausted retries for %s", cred_name, exc_info=True
            )
        return None

    def get(self, key: str) -> Optional[str]:
        self._load_secrets()
        return self._cache.get(key)

    def available(self) -> bool:
        try:
            import boto3  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            return False
        return any(k.startswith(self.KMS_PREFIX) for k in os.environ)


class FileBackend(CredentialBackend):
    """Reads credentials from a JSON file on disk (for air-gapped deployments).

    The file may be optionally base64-encoded (set ``CREDENTIAL_FILE_ENCODING=base64``).

    Configuration via environment variables:
    - ``CREDENTIAL_FILE_PATH``: Path to JSON credential file
    - ``CREDENTIAL_FILE_ENCODING``: ``plain`` (default) or ``base64``
    """

    name = "file"

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._loaded = False

    def _load_secrets(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        file_path = os.environ.get("CREDENTIAL_FILE_PATH", "")
        if not file_path:
            return
        path = Path(file_path)
        if not path.is_file():
            logger.warning("Credential file not found: %s", file_path)
            return
        try:
            raw = path.read_bytes()
            encoding = os.environ.get("CREDENTIAL_FILE_ENCODING", "plain").lower()
            if encoding == "base64":
                raw = base64.b64decode(raw)
            data = json.loads(raw)
            if isinstance(data, dict):
                self._cache = {k: str(v) for k, v in data.items()}
                logger.info("Loaded %d secrets from credential file %s", len(self._cache), file_path)
        except Exception:
            logger.warning("Failed to read credential file %s", file_path, exc_info=True)

    def get(self, key: str) -> Optional[str]:
        self._load_secrets()
        return self._cache.get(key)

    def available(self) -> bool:
        file_path = os.environ.get("CREDENTIAL_FILE_PATH", "")
        return bool(file_path) and Path(file_path).is_file()


# ---------------------------------------------------------------------------
# Placeholder / insecure-value detection
# ---------------------------------------------------------------------------


def is_placeholder(value: str) -> bool:
    """Return True if *value* looks like a placeholder or known-insecure default.

    Checks against:
    - Exact known-insecure values (``minioadmin``, ``password``, etc.)
    - Substring patterns (``change-me``, ``example``, ``your_``, etc.)
    - URL-embedded credentials (``amqp://user:password@host``)
    """
    if not value or not value.strip():
        return True
    lower = value.strip().lower()
    # Exact match
    if lower in PLACEHOLDER_VALUES:
        return True
    # Substring match
    for pattern in PLACEHOLDER_SUBSTRINGS:
        if pattern in lower:
            return True
    # Check URL-embedded credentials
    if "://" in value:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(value)
            if parsed.username and is_placeholder(parsed.username):
                return True
            if parsed.password and is_placeholder(parsed.password):
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class CredentialValidationResult:
    """Result of validating a single credential."""

    name: str
    status: str  # "ok", "missing", "empty", "placeholder", "not_required"
    message: str
    backend: str = ""


@dataclass
class ValidationReport:
    """Aggregate validation report for all credentials."""

    results: list[CredentialValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no credential has an error status."""
        return all(r.status in ("ok", "not_required") for r in self.results)

    @property
    def errors(self) -> list[CredentialValidationResult]:
        return [r for r in self.results if r.status not in ("ok", "not_required")]

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.status == "ok")
        total = len(self.results)
        status = "PASS" if self.passed else "FAIL"
        return f"{status}: {ok}/{total} credentials validated"


def _is_credential_required(schema_entry: dict[str, Any]) -> bool:
    """Determine if a credential is required in the current environment."""
    required_when = schema_entry.get("required_when", "optional")
    if required_when == "always":
        return True
    if required_when == "optional":
        return False
    # Conditional requirements like "STORAGE_BACKEND=s3"
    if "=" in required_when:
        env_var, expected = required_when.split("=", 1)
        actual = os.environ.get(env_var, "").strip().lower()
        return actual == expected.lower()
    return False


def validate_credentials(
    manager: Optional[CredentialManager] = None,
    strict: bool = True,
) -> ValidationReport:
    """Validate all credentials in the schema.

    Args:
        manager: CredentialManager instance (uses default if None).
        strict: If True, flag placeholder values as errors.

    Returns:
        ValidationReport with per-credential results.
    """
    if manager is None:
        manager = CredentialManager()
    report = ValidationReport()
    for name, schema_entry in CREDENTIAL_SCHEMA.items():
        required = _is_credential_required(schema_entry)
        value = manager.get(name)
        if value is None or value == "":
            if required:
                report.results.append(
                    CredentialValidationResult(
                        name=name,
                        status="missing",
                        message=f"Required credential '{name}' not found in any backend",
                    )
                )
            else:
                report.results.append(
                    CredentialValidationResult(
                        name=name,
                        status="not_required",
                        message=f"Optional credential '{name}' not set (acceptable)",
                    )
                )
        elif strict and is_placeholder(value):
            report.results.append(
                CredentialValidationResult(
                    name=name,
                    status="placeholder",
                    message=f"Credential '{name}' contains a placeholder value",
                )
            )
        else:
            backend_name = ""
            for backend in manager.backends:
                if backend.get(name) is not None:
                    backend_name = backend.name
                    break
            report.results.append(
                CredentialValidationResult(
                    name=name,
                    status="ok",
                    message=f"Credential '{name}' is set",
                    backend=backend_name,
                )
            )
    return report


# ---------------------------------------------------------------------------
# CredentialManager: unified retrieval with backend fallback chain
# ---------------------------------------------------------------------------


class CredentialManager:
    """Manages credential retrieval across multiple backends.

    Backends are tried in priority order. The first backend that returns
    a non-None value wins.

    Default backend order:
    1. Environment variables (always available)
    2. HashiCorp Vault (if ``hvac`` installed and configured)
    3. AWS Secrets Manager (if ``boto3`` installed and configured)
    4. AWS KMS (if ``boto3`` installed and ``KMS_ENC_*`` env vars present)
    5. Encrypted file (if ``CREDENTIAL_FILE_PATH`` is set)

    Credential rotation:
    If *refresh_interval_seconds* is set, the manager spawns a daemon thread
    that periodically re-fetches all cached credentials.  When a credential
    value changes, the optional *on_credential_refreshed* callback is invoked
    with the credential name.

    Example::

        manager = CredentialManager()
        password = manager.get("POSTGRES_PASSWORD")

        # With rotation callback:
        def _on_refresh(name: str) -> None:
            logging.info("Credential %s was rotated", name)

        manager = CredentialManager(
            refresh_interval_seconds=300,
            on_credential_refreshed=_on_refresh,
        )
    """

    def __init__(
        self,
        backends: Optional[list[CredentialBackend]] = None,
        refresh_interval_seconds: Optional[float] = None,
        on_credential_refreshed: Optional[Callable[[str], None]] = None,
    ) -> None:
        if backends is not None:
            self.backends = backends
        else:
            self.backends = self._default_backends()

        self._refresh_interval = refresh_interval_seconds
        self._on_credential_refreshed = on_credential_refreshed
        self._credential_cache: dict[str, Optional[str]] = {}
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self._refresh_interval is not None and self._refresh_interval > 0:
            self._start_refresh_thread()

    def _start_refresh_thread(self) -> None:
        """Launch a daemon thread that periodically re-fetches credentials."""
        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="credential-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        logger.info(
            "Credential refresh thread started (interval=%ss)",
            self._refresh_interval,
        )

    def _refresh_loop(self) -> None:
        """Background loop that re-fetches credentials at the configured interval."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._refresh_interval)
            if self._stop_event.is_set():
                break
            self._check_for_rotated_credentials()

    def _check_for_rotated_credentials(self) -> None:
        """Re-fetch all previously-accessed credentials and fire callback on change."""
        for key in list(self._credential_cache.keys()):
            try:
                new_value = self._fetch_from_backends(key)
                old_value = self._credential_cache.get(key)
                if new_value != old_value:
                    self._credential_cache[key] = new_value
                    logger.info("Credential '%s' value changed during refresh", key)
                    if self._on_credential_refreshed is not None:
                        try:
                            self._on_credential_refreshed(key)
                        except Exception:
                            logger.warning(
                                "on_credential_refreshed callback failed for %s",
                                key,
                                exc_info=True,
                            )
            except Exception:
                logger.warning(
                    "Failed to refresh credential %s", key, exc_info=True
                )

    def stop_refresh(self) -> None:
        """Stop the background credential refresh thread (if running)."""
        self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None
            logger.info("Credential refresh thread stopped")

    @staticmethod
    def _default_backends() -> list[CredentialBackend]:
        """Build the default backend chain.

        Order: env -> vault -> aws-secrets -> aws-kms -> file
        Only includes backends whose optional dependencies are importable.
        """
        backends: list[CredentialBackend] = [EnvVarBackend()]
        # Vault backend (lazy: only attempts import when get() is called)
        backends.append(VaultBackend())
        # AWS Secrets Manager backend
        backends.append(AWSSecretsBackend())
        # AWS KMS backend (for individually-encrypted credentials)
        backends.append(AWSKMSBackend())
        # File backend
        backends.append(FileBackend())
        return backends

    def _fetch_from_backends(self, key: str) -> Optional[str]:
        """Fetch a credential value from the backend chain (no caching)."""
        for backend in self.backends:
            try:
                value = backend.get(key)
                if value is not None:
                    return value
            except CredentialError:
                raise
            except Exception:
                logger.warning(
                    "Backend %s raised an exception for key %s",
                    backend.name,
                    key,
                    exc_info=True,
                )
        return None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a credential by name, trying backends in order.

        Args:
            key: Credential name (e.g. ``POSTGRES_PASSWORD``).
            default: Fallback value if no backend has the credential.

        Returns:
            The credential value, or *default* if not found.
        """
        value = self._fetch_from_backends(key)
        if value is not None:
            self._credential_cache[key] = value
            return value
        self._credential_cache[key] = default
        return default

    def get_required(self, key: str) -> str:
        """Retrieve a credential that must be present.

        Raises:
            LookupError: If the credential is not found in any backend.
        """
        value = self.get(key)
        if value is None:
            raise LookupError(
                f"Required credential '{key}' not found in any backend "
                f"(tried: {', '.join(b.name for b in self.backends)})"
            )
        return value

    def available_backends(self) -> list[str]:
        """Return the names of backends that report themselves as available."""
        return [b.name for b in self.backends if b.available()]


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

# Lazily initialized singleton
_default_manager: Optional[CredentialManager] = None


def get_credential(name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a credential by name using the default manager.

    This is the primary public API for credential retrieval. It tries
    backends in the default priority order (env -> vault -> aws -> file).

    Args:
        name: Credential name (e.g. ``POSTGRES_PASSWORD``).
        default: Fallback value if not found in any backend.

    Returns:
        The credential value, or *default* if not found.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = CredentialManager()
    return _default_manager.get(name, default)


def reset_default_manager() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _default_manager
    _default_manager = None
