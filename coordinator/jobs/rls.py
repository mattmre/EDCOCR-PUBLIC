"""Postgres row-level security tenant GUC helpers (R4-RC-07/08 / S2-A08).

Migration ``0015_jobs_rls_postgres`` enables FORCE RLS on ``jobs_job``.
Migration ``0016_jobs_child_rls_postgres`` extends FORCE RLS to job-child
tables (pages, custody, PII, extraction) via EXISTS join on ``jobs_job``.

Fail-closed policy: only rows whose parent ``jobs_job.tenant_id`` equals
``current_setting('app.tenant_id')`` are visible when GUC is non-empty.

Application code remains the primary tenant pin (ORM ``.filter(tenant_id=…)``).
This module is defense-in-depth so unscoped ORM/SQL cannot fleet-scan when a
tenant context is active.

Usage::

    from jobs.rls import tenant_rls_context, set_tenant_rls_context

    with tenant_rls_context("tenant-abc"):
        Job.objects.count()  # only that tenant's jobs under Postgres RLS

SQLite / non-Postgres backends are no-ops (local tests stay green).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Iterator, Optional

from django.db import connection, transaction

logger = logging.getLogger(__name__)

_GUC_NAME = "app.tenant_id"
_CONTROL_PLANE_GUC_NAME = "app.control_plane"


def _is_postgres() -> bool:
    return connection.vendor == "postgresql"


def _rls_set_config_fail_closed() -> bool:
    """Production (or explicit flag) refuses silent GUC set failures (C05)."""
    if os.environ.get("DEPLOYMENT_ENV", "").strip().lower() == "production":
        return True
    return os.environ.get("RLS_SET_CONFIG_FAIL_CLOSED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_tenant_rls_context(tenant_id: Optional[str], *, local: bool = True) -> None:
    """Set ``app.tenant_id`` for the current transaction or connection.

    Empty / None clears the GUC to '' which, under the fail-closed policy,
    yields zero job/child rows (no fleet leak). Callers that need cross-tenant
    fleet access must use a DB role with BYPASSRLS (operator-owned).

    When ``DEPLOYMENT_ENV=production`` or ``RLS_SET_CONFIG_FAIL_CLOSED=true``,
    a failed set_config raises (no swallow).
    """
    if not _is_postgres():
        return
    value = (tenant_id or "").strip()
    try:
        with connection.cursor() as cursor:
            # ``local=True`` is transaction-scoped. Worker tasks use the
            # connection-scoped form inside worker_tenant_rls_context so the
            # pin is established before the first ORM lookup and remains in
            # force across the task's short transactions and I/O.
            cursor.execute(
                "SELECT set_config(%s, %s, %s)",
                [_GUC_NAME, value, bool(local)],
            )
    except Exception as exc:
        logger.warning(
            "Failed to set %s=%r for RLS (is migration 0015/0016 applied?)",
            _GUC_NAME,
            value,
            exc_info=True,
        )
        if _rls_set_config_fail_closed():
            raise RuntimeError(
                f"RLS set_config failed for {_GUC_NAME} (C05 fail-closed)"
            ) from exc


def clear_tenant_rls_context() -> None:
    """Clear tenant GUC (fail-closed: no job rows under FORCE RLS)."""
    set_tenant_rls_context("", local=False)


@contextmanager
def tenant_rls_context(tenant_id: Optional[str]) -> Iterator[None]:
    """Context manager: set tenant GUC for the block, then clear."""
    with transaction.atomic():
        set_tenant_rls_context(tenant_id)
        try:
            yield
        finally:
            clear_tenant_rls_context()


@contextmanager
def worker_tenant_rls_context(tenant_id: Optional[str]) -> Iterator[None]:
    """Pin a worker connection to one tenant for the complete task.

    Celery tasks perform many short ORM transactions around storage I/O. A
    ``SET LOCAL`` issued before the first transaction disappears immediately
    and leaves later writes unpinned; holding one database transaction over all
    worker I/O would create long-lived locks. This task-scoped connection pin
    is the safe middle ground: establish it before the job lookup, restore the
    previous value on exit, and fail closed when production omitted the
    persisted tenant argument.

    SQLite remains a no-op for the local test harness.
    """
    if not _is_postgres():
        yield
        return

    value = (tenant_id or "").strip()
    if not value:
        # Legacy/single-tenant coordinator jobs intentionally persist a blank
        # tenant.  They are not addressable through a tenant policy, so use
        # the explicit control-plane branch for the bounded signed task.
        with control_plane_rls_context():
            yield
        return
    previous = get_tenant_rls_context()
    set_tenant_rls_context(value, local=False)
    try:
        if get_tenant_rls_context() != value:
            raise RuntimeError("worker tenant RLS binding was not applied")
        yield
    finally:
        # Restoring the previous value is important for pooled Celery worker
        # connections. A blank value is the fail-closed idle state.
        set_tenant_rls_context(previous, local=False)


def get_tenant_rls_context() -> str:
    """Return current app.tenant_id GUC (empty if unset / non-Postgres)."""
    if not _is_postgres():
        return ""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_setting(%s, true)",
                [_GUC_NAME],
            )
            row = cursor.fetchone()
            return (row[0] if row and row[0] is not None else "") or ""
    except Exception:
        return ""


def get_control_plane_rls_context() -> str:
    """Return the current control-plane GUC (empty off/non-Postgres)."""
    if not _is_postgres():
        return ""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_setting(%s, true)", [_CONTROL_PLANE_GUC_NAME]
            )
            row = cursor.fetchone()
            return (row[0] if row and row[0] is not None else "") or ""
    except Exception:
        return ""


@contextmanager
def control_plane_rls_context() -> Iterator[None]:
    """Grant one trusted worker operation explicit cross-tenant visibility.

    FORCE-RLS tables used by the durable relay and expiry sweep cannot be
    enumerated under a tenant GUC.  The migration installs a narrowly named
    policy branch for this GUC; callers must opt in for the smallest possible
    block and the prior connection value is restored for pooled workers.
    """
    if not _is_postgres():
        yield
        return

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_setting(%s, true)", [_CONTROL_PLANE_GUC_NAME]
            )
            row = cursor.fetchone()
            previous = (row[0] if row and row[0] is not None else "") or ""
            cursor.execute(
                "SELECT set_config(%s, %s, false)",
                [_CONTROL_PLANE_GUC_NAME, "on"],
            )
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config(%s, %s, false)",
                [_CONTROL_PLANE_GUC_NAME, previous],
            )


def tenant_rls_command(func):
    """Decorate a Django command handler with its validated tenant scope."""

    @wraps(func)
    def wrapped(self, *args, **options):
        tenant_id = str(options.get("tenant_id") or "").strip()
        if not tenant_id:
            return func(self, *args, **options)
        with tenant_command_rls_context(tenant_id):
            return func(self, *args, **options)

    return wrapped


@contextmanager
def tenant_command_rls_context(tenant_id: str) -> Iterator[None]:
    """Connection-pin a command without wrapping storage I/O in one transaction."""
    if not _is_postgres():
        yield
        return
    value = str(tenant_id or "").strip()
    if not value:
        raise RuntimeError("tenant command RLS context requires a tenant_id")
    previous = get_tenant_rls_context()
    set_tenant_rls_context(value, local=False)
    try:
        if get_tenant_rls_context() != value:
            raise RuntimeError("tenant command RLS binding was not applied")
        yield
    finally:
        set_tenant_rls_context(previous, local=False)


def control_plane_rls_command(func):
    """Decorate a trusted fleet-wide Django command handler for FORCE RLS."""

    @wraps(func)
    def wrapped(self, *args, **options):
        with control_plane_rls_context():
            return func(self, *args, **options)

    return wrapped
