"""HMAC task authenticity for Celery messages (R2-RC-01).

Broker access alone must not be equivalent to full pipeline authority.
When signing is enabled, every task message carries a short-lived HMAC in a
message header, bound to the task name and primary resource id (usually
``job_id``). The envelope is deliberately not a task kwarg: Celery validates
declared task signatures before publish, so an internal kwarg would make
strict-signature periodic tasks undispatchable.

Environment
-----------
``CELERY_TASK_SIGNING_KEY``
    Shared secret used to MAC task payloads. Required when signing is
    required (production or ``ENABLE_CELERY_TASK_SIGNING=true``).

``ENABLE_CELERY_TASK_SIGNING``
    Opt-in for non-production environments (``1``/``true``/``yes``).

``DEPLOYMENT_ENV=production``
    Fail-closed: signing is required; missing key refuses tasks.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_AUTH_KW = "_task_auth"
_DEFAULT_TTL_SECONDS = 3600
_MAX_SKEW_SECONDS = 60


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def get_signing_key() -> str:
    """Return the configured MAC secret (empty if unset)."""
    return os.environ.get("CELERY_TASK_SIGNING_KEY", "").strip()


def signing_enabled() -> bool:
    """True when a signing key is configured (verify + sign on dispatch)."""
    return bool(get_signing_key())


def signing_required() -> bool:
    """True when unsigned tasks must be rejected."""
    if os.environ.get("DEPLOYMENT_ENV", "").strip().lower() == "production":
        return True
    return _env_truthy("ENABLE_CELERY_TASK_SIGNING")


def extract_binding(task_name: str, args: tuple, kwargs: dict) -> str:
    """Derive a stable binding id for the MAC.

    Prefer explicit ``job_id`` / ``batch_id`` kwargs, then first string arg
    (most pipeline tasks pass ``job_id`` as args[0]), else task name alone
    (beat / fleet tasks without a resource id).
    """
    if kwargs:
        for key in ("job_id", "batch_id", "document_id"):
            val = kwargs.get(key)
            if isinstance(val, str) and val:
                return val
    if args:
        first = args[0]
        if isinstance(first, str) and first:
            return first
    return task_name


def _kwargs_payload_digest(kwargs: Optional[dict]) -> str:
    """Stable digest of kwargs excluding auth envelope (S2-A01 kwargs MAC)."""
    if not kwargs:
        return ""
    safe = {k: v for k, v in kwargs.items() if k != _AUTH_KW}
    # Prefer JSON; fall back to sorted repr for non-JSON values.
    try:
        import json

        blob = json.dumps(safe, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        blob = repr(sorted(safe.items(), key=lambda kv: str(kv[0])))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _args_payload_digest(args: Optional[tuple]) -> str:
    """Stable digest of positional args (A03 / Sweep A — args not previously MAC'd)."""
    if not args:
        return ""
    try:
        import json

        blob = json.dumps(list(args), default=str, separators=(",", ":"))
    except Exception:
        blob = repr(tuple(args))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def sign_task(
    task_name: str,
    binding: str,
    *,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> dict[str, Any]:
    """Build a ``_task_auth`` dict for kwargs.

    v3 MAC binds task name, binding, exp, kwargs digest, **and args digest**
    so positional-arg swap after signing is detected (Sweep A A03).
    """
    key = get_signing_key()
    if not key:
        raise RuntimeError("CELERY_TASK_SIGNING_KEY is required to sign Celery tasks")
    exp = int(time.time()) + max(60, int(ttl_seconds))
    payload = _kwargs_payload_digest(kwargs)
    args_payload = _args_payload_digest(args)
    msg = f"v3|{task_name}|{binding}|{exp}|{payload}|{args_payload}".encode("utf-8")
    sig = hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return {
        "v": 3,
        "exp": exp,
        "sig": sig,
        "binding": binding,
        "payload": payload,
        "args_payload": args_payload,
    }


def verify_task(
    task_name: str,
    binding: str,
    auth: Optional[dict],
    *,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> None:
    """Raise ``PermissionError`` if auth is missing/invalid when required."""
    required = signing_required()
    enabled = signing_enabled()

    if not required and not enabled:
        return

    if required and not enabled:
        raise PermissionError(
            "Celery task signing is required "
            "(set CELERY_TASK_SIGNING_KEY; DEPLOYMENT_ENV=production or "
            "ENABLE_CELERY_TASK_SIGNING=true)"
        )

    if not isinstance(auth, dict):
        raise PermissionError("Missing Celery task authenticity token")

    try:
        exp = int(auth.get("exp", 0))
        sig = str(auth.get("sig") or "")
        version = int(auth.get("v", 0))
    except (TypeError, ValueError) as exc:
        raise PermissionError("Malformed Celery task authenticity token") from exc

    if version not in (1, 2, 3) or not sig:
        raise PermissionError("Unsupported Celery task authenticity version")

    # Post-sweep HIGH: production / CELERY_TASK_SIGNING_MIN_VERSION refuse
    # legacy v1/v2 tokens (arg-swap weaker digests). Lab may still accept
    # in-flight v1/v2 unless min version is raised.
    min_ver = 1
    try:
        min_ver = int(os.environ.get("CELERY_TASK_SIGNING_MIN_VERSION", "0") or "0")
    except (TypeError, ValueError):
        min_ver = 0
    dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    if dep in ("production", "prod") and min_ver < 3:
        min_ver = 3
    if min_ver > 0 and version < min_ver:
        raise PermissionError(
            f"Celery task authenticity version v{version} below minimum v{min_ver}"
        )

    now = int(time.time())
    if exp < now - _MAX_SKEW_SECONDS:
        raise PermissionError("Celery task authenticity token expired")

    # Prefer binding embedded in token when present (defends arg swap).
    token_binding = auth.get("binding")
    if isinstance(token_binding, str) and token_binding:
        if token_binding != binding:
            raise PermissionError("Celery task binding mismatch")
        binding = token_binding

    key = get_signing_key()
    if version == 1:
        # Backward-compatible verify for in-flight v1 tokens (lab only).
        msg = f"v1|{task_name}|{binding}|{exp}".encode("utf-8")
    elif version == 2:
        payload = _kwargs_payload_digest(kwargs)
        token_payload = str(auth.get("payload") or "")
        if token_payload and token_payload != payload:
            raise PermissionError("Celery task kwargs payload mismatch")
        msg = f"v2|{task_name}|{binding}|{exp}|{payload}".encode("utf-8")
    else:
        # v3: kwargs + positional args digests (A03).
        payload = _kwargs_payload_digest(kwargs)
        args_payload = _args_payload_digest(args)
        token_payload = str(auth.get("payload") or "")
        token_args = str(auth.get("args_payload") or "")
        if token_payload and token_payload != payload:
            raise PermissionError("Celery task kwargs payload mismatch")
        if token_args and token_args != args_payload:
            raise PermissionError("Celery task args payload mismatch")
        msg = f"v3|{task_name}|{binding}|{exp}|{payload}|{args_payload}".encode("utf-8")
    expected = hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise PermissionError("Invalid Celery task authenticity signature")


def attach_auth(
    task_name: str,
    args: Optional[tuple],
    kwargs: Optional[dict],
) -> dict:
    """Return kwargs with ``_task_auth`` attached when signing is enabled."""
    kwargs = dict(kwargs or {})
    if not signing_enabled():
        if signing_required():
            raise RuntimeError(
                "CELERY_TASK_SIGNING_KEY required for signed Celery dispatch"
            )
        return kwargs
    args_t = tuple(args or ())
    binding = extract_binding(task_name, args_t, kwargs)
    kwargs[_AUTH_KW] = sign_task(task_name, binding, args=args_t, kwargs=kwargs)
    return kwargs


def attach_auth_header(
    task_name: str,
    args: Optional[tuple],
    kwargs: Optional[dict],
    headers: Optional[dict],
) -> dict:
    """Return message headers carrying a fresh authenticity envelope."""
    headers = dict(headers or {})
    if not signing_enabled():
        if signing_required():
            raise RuntimeError(
                "CELERY_TASK_SIGNING_KEY required for signed Celery dispatch"
            )
        return headers
    args_t = tuple(args or ())
    kwargs_d = dict(kwargs or {})
    binding = extract_binding(task_name, args_t, kwargs_d)
    headers[_AUTH_KW] = sign_task(
        task_name,
        binding,
        args=args_t,
        kwargs=kwargs_d,
    )
    return headers


def pop_and_verify(
    task_name: str,
    args: tuple,
    kwargs: dict,
) -> dict:
    """Pop ``_task_auth`` and verify; return remaining kwargs."""
    kwargs = dict(kwargs or {})
    auth = kwargs.pop(_AUTH_KW, None)
    binding = extract_binding(task_name, args, kwargs)
    verify_task(
        task_name,
        binding,
        auth if isinstance(auth, dict) else None,
        args=args,
        kwargs=kwargs,
    )
    return kwargs


# ---------------------------------------------------------------------------
# Celery Task base
# ---------------------------------------------------------------------------

try:
    from celery import Task
except ImportError:  # pragma: no cover - coordinator image always has celery
    Task = object  # type: ignore[misc, assignment]


class SignedTask(Task):  # type: ignore[misc, valid-type]
    """Celery Task base that MAC-signs on dispatch and verifies on receive.

    When signing is disabled (no key, non-production), behaviour matches the
    stock Task class. When a key is configured, every ``apply_async`` attaches
    ``_task_auth`` as a message header and every execution verifies it before
    ``run``. Legacy in-flight kwarg envelopes remain readable during rollout.
    """

    abstract = True
    # The signed envelope is intentionally not part of application task
    # signatures. Validate the caller's original arguments here, then disable
    # Celery's second validation pass after the private envelope is attached.
    typing = False

    def apply_async(self, args=None, kwargs=None, **options):
        name = getattr(self, "name", None) or self.__name__
        check_arguments = getattr(self, "__header__", None)
        if check_arguments is not None:
            check_arguments(*(args or ()), **(kwargs or {}))
        try:
            options["headers"] = attach_auth_header(
                name,
                args,
                kwargs,
                options.get("headers"),
            )
        except RuntimeError:
            logger.error(
                "Refusing to dispatch %s without CELERY_TASK_SIGNING_KEY", name
            )
            raise
        return super().apply_async(args=args, kwargs=kwargs, **options)

    def __call__(self, *args, **kwargs):
        name = getattr(self, "name", None) or self.__name__
        try:
            kwargs = dict(kwargs or {})
            legacy_auth = kwargs.pop(_AUTH_KW, None)
            request_headers = getattr(self.request, "headers", None) or {}
            header_auth = (
                request_headers.get(_AUTH_KW)
                if isinstance(request_headers, dict)
                else None
            )
            if legacy_auth is not None and header_auth is not None:
                raise PermissionError("Ambiguous Celery task authenticity token")
            auth = header_auth if header_auth is not None else legacy_auth
            binding = extract_binding(name, args, kwargs)
            verify_task(
                name,
                binding,
                auth if isinstance(auth, dict) else None,
                args=args,
                kwargs=kwargs,
            )
        except PermissionError as exc:
            logger.error("Celery task authenticity failed for %s: %s", name, exc)
            raise
        return super().__call__(*args, **kwargs)
