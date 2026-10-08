"""RC-05 / R2-RC-01 residual: refuse bootstrap/admin broker principals in prod.

Product path enforces that ``CELERY_BROKER_URL`` does not embed administrator
or default-guest RabbitMQ users when deployment is production (or when
``REQUIRE_BROKER_LEAST_PRIV`` is explicitly enabled).

Least-privilege principals (see ``scripts/setup_rabbitmq_acls.py``):

* ``ocr_publisher`` — Django / celery-coordinator / beat publish path
* ``ocr_worker`` — worker consume path

Bootstrap ``ocr_user`` / ``ocr_admin`` remain valid for operator management
API only and must not appear in runtime Celery broker URLs under production.
"""

from __future__ import annotations

import os
import re
from typing import FrozenSet, Optional
from urllib.parse import unquote, urlparse

# Principals that must never be used as CELERY_BROKER_URL user in production.
DEFAULT_DENIED_BROKER_USERS: FrozenSet[str] = frozenset(
    {
        "guest",
        "admin",
        "rabbitmq",
        "ocr_admin",
        "ocr_user",  # farm bootstrap administrator
        "root",
    }
)

# Recommended least-priv set (not hard-required; deny-list is the fail-closed gate).
DEFAULT_ALLOWED_BROKER_USERS: FrozenSet[str] = frozenset(
    {
        "ocr_publisher",
        "ocr_worker",
    }
)


class BrokerLeastPrivError(ValueError):
    """Raised when CELERY_BROKER_URL violates least-privilege policy."""


def broker_user_from_url(url: str) -> Optional[str]:
    """Extract username from an AMQP/Redis broker URL (no password logged)."""
    raw = (url or "").strip()
    if not raw:
        return None
    # urlparse handles amqp://user:pass@host/vhost  # pragma: allowlist secret
    parsed = urlparse(raw)
    if parsed.username:
        return unquote(parsed.username).casefold()
    # Fallback for unusual encodings
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^:/@]+)[:@]", raw)
    if m:
        return unquote(m.group(1)).casefold()
    return None


def least_priv_enforced() -> bool:
    """True when production or explicit REQUIRE_BROKER_LEAST_PRIV."""
    if os.environ.get("REQUIRE_BROKER_LEAST_PRIV", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    return dep in ("production", "prod")


def denied_broker_users() -> FrozenSet[str]:
    extra = os.environ.get("BROKER_LEAST_PRIV_DENIED_USERS", "")
    denied = {u.casefold() for u in DEFAULT_DENIED_BROKER_USERS}
    for part in extra.split(","):
        p = part.strip().casefold()
        if p:
            denied.add(p)
    # Optional allowlist override can remove a denied entry for break-glass
    allow = os.environ.get("BROKER_LEAST_PRIV_ALLOW_DENIED", "")
    for part in allow.split(","):
        p = part.strip().casefold()
        if p and p in denied:
            denied.discard(p)
    return frozenset(denied)


def check_broker_url_least_priv(
    url: str,
    *,
    enforce: Optional[bool] = None,
    role: str = "runtime",
) -> Optional[str]:
    """Return error message if *url* violates policy; else None.

    When ``enforce`` is False (or None and least_priv_enforced() is False),
    always returns None (lab/dev remains usable with bootstrap users).
    """
    if enforce is None:
        enforce = least_priv_enforced()
    if not enforce:
        return None
    user = broker_user_from_url(url)
    if not user:
        # memory:// and similar non-AMQP backends have no user — ok
        if "://" in (url or "") and not urlparse(url).username:
            scheme = urlparse(url).scheme.lower()
            if scheme in (
                "memory",
                "redis",
                "rediss",
                "sentinel",
                "sqla",
                "db+",
                "cache",
            ):
                return None
        return (
            f"CELERY_BROKER_URL ({role}) has no username under least-priv enforcement "
            "(RC-05). Use ocr_publisher or ocr_worker principals."
        )
    if user in denied_broker_users():
        return (
            f"CELERY_BROKER_URL ({role}) user {user!r} is denied under "
            "broker least-privilege policy (RC-05 / R2-RC-01 residual). "
            "Use ocr_publisher (coordinator/publish) or ocr_worker (workers); "
            "bootstrap ocr_user/ocr_admin are management-only."
        )
    return None


def assert_broker_url_least_priv(url: str, *, role: str = "runtime") -> None:
    """Raise BrokerLeastPrivError when policy is violated under enforcement."""
    err = check_broker_url_least_priv(url, role=role)
    if err:
        raise BrokerLeastPrivError(err)
