"""Tenant lifecycle management — CRUD, API key management, resolution."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import secrets
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from api.database import (
    Batch,
    Tenant,
    TenantApiKey,
    TenantRateLimit,
    get_session_factory,
)

logger = logging.getLogger(__name__)

PURGE_FAILED_STATUS = "purge_failed"
_PURGE_SCAN_BATCH_SIZE = 1_000
_PURGE_LEDGER_MAX_LINE_BYTES = 64 * 1024
_PURGE_LEDGER_LOCK = threading.Lock()


def _tenant_purge_ledger_path() -> Path:
    from api import config

    configured = os.environ.get("TENANT_PURGE_LEDGER_PATH", "").strip()
    if configured:
        return Path(configured)
    output_root = Path(config.OUTPUT_FOLDER)
    # The container default is /app/ocr_output. A source checkout has no /app;
    # keep local evidence under the repository's ordinary ocr_output volume.
    if output_root.is_absolute() and not output_root.parent.exists():
        output_root = Path.cwd() / "ocr_output"
    return output_root / "logs" / "tenant-purge-custody.jsonl"


def _read_tenant_purge_ledger_state(tenant_id: str) -> tuple[str, dict | None]:
    """Verify the retained ledger and return its head plus this tenant's tail."""
    path = _tenant_purge_ledger_path()
    if not path.exists():
        return "", None
    previous_hash = ""
    tenant_tail = None
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline(_PURGE_LEDGER_MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line.encode("utf-8")) > _PURGE_LEDGER_MAX_LINE_BYTES:
                raise RuntimeError("tenant purge ledger contains an oversized event")
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("prev_hash", "") != previous_hash:
                raise RuntimeError("tenant purge ledger hash chain is broken")
            claimed_hash = str(event.get("hash", ""))
            hashable = {key: value for key, value in event.items() if key != "hash"}
            computed = hashlib.sha256(
                json.dumps(hashable, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if claimed_hash != computed:
                raise RuntimeError("tenant purge ledger event hash is invalid")
            previous_hash = claimed_hash
            if event.get("tenant_id") == tenant_id:
                tenant_tail = event
    return previous_hash, tenant_tail


def _record_tenant_purge_event(
    *, tenant_id: str, purge_id: str, event_type: str, data: dict | None = None
) -> dict:
    """Append and fsync one jobless, tamper-evident purge custody event."""
    path = _tenant_purge_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _PURGE_LEDGER_LOCK:
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                previous_hash, _ = _read_tenant_purge_ledger_state(tenant_id)
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                    "tenant_id": tenant_id,
                    "purge_id": purge_id,
                    "event_type": event_type,
                    "data": data or {},
                    "prev_hash": previous_hash,
                }
                event["hash"] = hashlib.sha256(
                    json.dumps(event, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                encoded = json.dumps(event, sort_keys=True, default=str)
                if len(encoded.encode("utf-8")) > _PURGE_LEDGER_MAX_LINE_BYTES:
                    raise RuntimeError("tenant purge custody event exceeds size bound")
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return event


def _get_tenant_purge_tail(tenant_id: str) -> dict | None:
    """Read a stable verified ledger tail while excluding other appenders."""
    path = _tenant_purge_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _PURGE_LEDGER_LOCK:
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                _, tail = _read_tenant_purge_ledger_state(tenant_id)
                return tail
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _utcnow() -> datetime:
    """Return current UTC time without tzinfo (matches project convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _generate_tenant_id() -> str:
    """Generate a unique tenant identifier: tenant_{hex12}."""
    return f"tenant_{uuid.uuid4().hex[:12]}"


def _generate_key_id() -> str:
    """Generate a unique API key identifier: key_{hex12}."""
    return f"key_{uuid.uuid4().hex[:12]}"


def _generate_api_key() -> str:
    """Generate a random 48-character API key: ocr_{urlsafe44}."""
    return f"ocr_{secrets.token_urlsafe(33)}"


def hash_api_key(raw_key: str) -> str:
    """Compute SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@contextmanager
def _admission_serialization():
    """Serialize tenant lifecycle changes with job admission.

    A tenant status/key mutation must share the same process and cross-process
    lock as ``JobManager.submit``.  Otherwise a submitter can read an active
    tenant, a purge can revoke/delete it, and the submitter can commit a job
    after the purge's resource snapshot.  The import stays local so the
    tenant manager does not participate in the job-manager module import
    cycle.
    """
    from api.job_manager import (
        _ADMIT_CAPACITY_LOCK,
        _acquire_admit_capacity_lock,
        _cross_process_admit_lock,
    )

    with _ADMIT_CAPACITY_LOCK:
        with _cross_process_admit_lock():
            # The caller's session is supplied by the surrounding operation;
            # the DB lock is acquired by that operation immediately after
            # entering this context.
            yield _acquire_admit_capacity_lock


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------


def create_tenant(
    *,
    name: str,
    display_name: Optional[str] = None,
    tier: str = "standard",
    max_concurrent_jobs: int = 4,
    max_pages_per_month: int = 10000,
    max_storage_bytes: int = 10 * 1024**3,
    allowed_features: Optional[list[str]] = None,
    admin_email: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tenant:
    """Create a new tenant record and return it."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        tenant = Tenant(
            tenant_id=_generate_tenant_id(),
            name=name,
            display_name=display_name,
            status="active",
            tier=tier,
            created_at=_utcnow(),
            max_concurrent_jobs=max_concurrent_jobs,
            max_pages_per_month=max_pages_per_month,
            max_storage_bytes=max_storage_bytes,
            allowed_features=json.dumps(allowed_features or []),
            admin_email=admin_email,
        )
        session.add(tenant)
        session.commit()
        session.refresh(tenant)
        logger.info("Created tenant %s (%s)", tenant.tenant_id, name)
        return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def get_tenant(
    tenant_id: str, *, session: Optional[Session] = None
) -> Optional[Tenant]:
    """Return a tenant by ID, or None if not found."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        return session.get(Tenant, tenant_id)
    finally:
        if own_session:
            session.close()


def list_tenants(
    status_filter: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    session: Optional[Session] = None,
) -> list[Tenant]:
    """Return tenants, optionally filtered and paginated.

    ``limit=None`` preserves the internal caller contract.  Public admin
    routes pass an explicit bounded limit so tenant cardinality cannot turn a
    single control-plane response into an unbounded ORM/materialization job.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        q = session.query(Tenant)
        if status_filter:
            q = q.filter(Tenant.status == status_filter)
        q = q.order_by(Tenant.created_at.desc())
        if offset:
            q = q.offset(offset)
        if limit is not None:
            q = q.limit(limit)
        return q.all()
    finally:
        if own_session:
            session.close()


def update_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
    **kwargs,
) -> Optional[Tenant]:
    """Update tenant fields. Returns updated tenant or None if not found."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        with _admission_serialization() as acquire_db_lock:
            acquire_db_lock(session)
            tenant = session.get(Tenant, tenant_id)
            if not tenant:
                return None

            allowed_fields = {
                "name",
                "display_name",
                "status",
                "tier",
                "max_concurrent_jobs",
                "max_pages_per_month",
                "max_storage_bytes",
                "allowed_features",
                "admin_email",
            }
            for key, value in kwargs.items():
                if key not in allowed_fields:
                    continue
                if (
                    key == "status"
                    and value == "active"
                    and tenant.status == PURGE_FAILED_STATUS
                ):
                    raise ValueError(
                        "Tenant purge recovery must complete before activation"
                    )
                if key == "allowed_features" and isinstance(value, list):
                    value = json.dumps(value)
                setattr(tenant, key, value)

            tenant.updated_at = _utcnow()
            session.commit()
            session.refresh(tenant)
            logger.info("Updated tenant %s", tenant_id)
            return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def suspend_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Set tenant status to 'suspended'."""
    return update_tenant(tenant_id, session=session, status="suspended")


def activate_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Set tenant status to 'active'."""
    return update_tenant(tenant_id, session=session, status="active")


def delete_tenant(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[Tenant]:
    """Soft-delete a tenant and revoke all associated API keys.

    Sets tenant status to 'deleted' (data is preserved for audit trail)
    and revokes every active API key belonging to the tenant.
    Returns the updated tenant, or None if the tenant was not found.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        with _admission_serialization() as acquire_db_lock:
            acquire_db_lock(session)
            tenant = session.get(Tenant, tenant_id)
            if not tenant:
                return None

            tenant.status = "deleted"
            tenant.updated_at = _utcnow()

            # Revoke all active keys for this tenant
            active_keys = (
                session.query(TenantApiKey)
                .filter(
                    TenantApiKey.tenant_id == tenant_id,
                    TenantApiKey.status == "active",
                )
                .all()
            )
            revoked_count = 0
            for key in active_keys:
                key.status = "revoked"
                revoked_count += 1

            session.commit()
            session.refresh(tenant)
            logger.info(
                "Soft-deleted tenant %s, revoked %d API keys",
                tenant_id,
                revoked_count,
            )
            return tenant
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Hard purge: cascading deletion of all tenant data
# ---------------------------------------------------------------------------


# Back-compat re-exports for tests/callers that imported private helpers.
from ocr_distributed import tenant_purge_fs as _tenant_purge_fs  # noqa: E402

_path_based_doc_id = _tenant_purge_fs.path_based_doc_id
_compute_source_fingerprint_fast = _tenant_purge_fs.compute_source_fingerprint_fast
_build_resume_doc_id = _tenant_purge_fs.build_resume_doc_id
_shared_purge_fs_job_artifacts = _tenant_purge_fs.purge_fs_job_artifacts


def _purge_fs_job_artifacts(
    *,
    job_id: str,
    result_path: Optional[str],
    tenant_id: str,
    config_mod,
    source_file: Optional[str] = None,
    source_hash: Optional[str] = None,
) -> int:
    """Remove job FS trees under SOURCE/OUTPUT/TEMP/EXPORT (R4-RC-16 / PB-A01).

    Shared with coordinator ``manage.py purge_tenant`` via
    ``ocr_distributed.tenant_purge_fs`` so both planes wipe the same FS classes.
    """
    return _shared_purge_fs_job_artifacts(
        job_id=job_id,
        tenant_id=tenant_id,
        source_file=source_file,
        source_hash=source_hash,
        result_path=result_path,
        config_mod=config_mod,
        strict=True,
    )


def _purge_auxiliary_api_data(
    tenant_id: str,
    resource_ids: set[str],
) -> dict[str, int]:
    """Purge non-ORM API data planes for a tenant.

    Replay/index/review data lives in separate SQLite stores and analytics /
    dashboard history is process-local.  None is covered by the SQLAlchemy
    tenant cascade.  Existing stores are reused; an unused optional store is
    not created merely because a tenant is being deleted.  Exceptions
    intentionally propagate so a caller cannot report success while a data
    plane remains.
    """
    counts = {
        "event_store_events": 0,
        "entity_index_records": 0,
        "review_queue_records": 0,
        "analytics_records": 0,
        "dashboard_records": 0,
    }

    from api import event_store

    event_instance = event_store._store
    close_event = False
    if event_instance is None:
        event_path = event_store._config_value(
            "EVENT_STORE_PATH", event_store._DEFAULT_DB_PATH
        )
        if Path(event_path).exists():
            event_instance = event_store.EventStore(db_path=str(event_path))
            close_event = True
    if event_instance is not None:
        try:
            counts["event_store_events"] = event_instance.delete_for_resources(
                resource_ids
            )
        finally:
            if close_event:
                event_instance.close()

    from api import entity_index

    index_instance = None
    try:
        from api.routers import recall

        index_instance = recall._entity_index
    except Exception:
        # Recall is optional in minimal API/test contexts.
        index_instance = None
    close_index = False
    if index_instance is None and Path(entity_index._DEFAULT_DB_PATH).exists():
        index_instance = entity_index.EntityIndex(db_path=entity_index._DEFAULT_DB_PATH)
        close_index = True
    if index_instance is not None:
        try:
            counts["entity_index_records"] = index_instance.remove_jobs(resource_ids)
        finally:
            if close_index:
                index_instance.close()

    from api import review_queue

    review_instance = None
    try:
        from api.routers import review

        review_instance = review._review_queue
    except Exception:
        review_instance = None
    close_review = False
    if review_instance is None and Path(review_queue.DB_PATH).exists():
        review_instance = review_queue.ReviewQueue(db_path=review_queue.DB_PATH)
        close_review = True
    if review_instance is not None:
        try:
            removed = review_instance.remove_jobs(resource_ids)
            if hasattr(review_instance, "remove_tenant_metadata"):
                removed += review_instance.remove_tenant_metadata(tenant_id)
            counts["review_queue_records"] = removed
        finally:
            if close_review:
                review_instance.close()

    from api import analytics

    if analytics._store is not None:
        counts["analytics_records"] = analytics._store.purge_tenant(tenant_id)

    from api import dashboard

    if dashboard._collector is not None:
        counts["dashboard_records"] = dashboard._collector.purge_tenant(tenant_id)

    return counts


def purge_tenant_data(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
    include_fs: bool = True,
) -> Optional[dict]:
    """Hard-delete API-plane data for a tenant and remove its API row.

    Deletes API Jobs, batches, UsageRecords, TenantApiKeys, TenantRateLimit,
    webhook-DLQ entries, auxiliary replay/index/review stores, in-memory
    analytics/dashboard records, and the Tenant row itself.  The coordinator's
    separate Django/PostgreSQL data plane must be purged independently.
    R4-RC-16: when *include_fs* is True, also removes job source/output/temp
    and EXPORT artifact directories (not only SOURCE/OUTPUT job_id dirs).

    Corrupt-row policy (skip-with-signal): legacy batches whose settings
    blob is unreadable are EXCLUDED from the purge snapshot, each exclusion
    emits a ``logger.warning`` with batch id + tenant, and the total is
    exposed as ``legacy_batches_skipped`` in the returned ``deleted`` map.
    """
    from ocr_distributed.litigation_hold import is_litigation_hold_active

    if is_litigation_hold_active():
        raise RuntimeError(
            "LITIGATION_HOLD is active -- all tenant data deletions are blocked."
        )

    own_session = session is None
    if own_session:
        session = get_session_factory()()
    fenced = False
    sql_committed = False
    tail = _get_tenant_purge_tail(tenant_id)
    if tail and tail.get("event_type") != "tenant_purge_completed":
        purge_id = str(tail.get("purge_id") or uuid.uuid4())
    else:
        purge_id = str(uuid.uuid4())
    try:
        # Fence admission before changing status and taking the resource
        # snapshot.  The commit is deliberately short: the filesystem and
        # auxiliary stores are purged after the fence is visible, so later
        # submissions/key creation fail closed without blocking every tenant
        # for the duration of a large purge.
        with _admission_serialization() as acquire_db_lock:
            acquire_db_lock(session)
            tenant = session.get(Tenant, tenant_id)
            if not tenant:
                if tail and (
                    tail.get("event_type") == "tenant_purge_sql_delete_started"
                    or (
                        tail.get("event_type") == "tenant_purge_failed"
                        and bool((tail.get("data") or {}).get("sql_committed"))
                    )
                ):
                    _record_tenant_purge_event(
                        tenant_id=tenant_id,
                        purge_id=purge_id,
                        event_type="tenant_purge_completed",
                        data={"reconciled_after_sql_commit": True},
                    )
                    return {
                        "tenant_id": tenant_id,
                        "deleted": {},
                        "reconciled_after_sql_commit": True,
                    }
                return None
            resumed = tenant.status in {"deleting", PURGE_FAILED_STATUS}
            tenant.status = "deleting"
            tenant.updated_at = _utcnow()
            active_keys = (
                session.query(TenantApiKey)
                .filter(
                    TenantApiKey.tenant_id == tenant_id,
                    TenantApiKey.status == "active",
                )
                .all()
            )
            for key in active_keys:
                key.status = "revoked"
            session.commit()
            fenced = True
            _record_tenant_purge_event(
                tenant_id=tenant_id,
                purge_id=purge_id,
                event_type=(
                    "tenant_purge_resumed" if resumed else "tenant_purge_started"
                ),
                data={"include_fs": include_fs},
            )

        from api import config
        from api.database import Job, UsageRecord
        from api.job_manager import get_manager

        # Admission is fenced, but already-running subprocesses can still
        # recreate source/output bytes after a filesystem sweep. Cancel every
        # tenant job and wait for the local writer thread to finish before the
        # resource snapshot. The API enforces a single process/pod topology,
        # so this joins the complete shipped execution plane.
        quiesce_ids = [
            str(row[0])
            for row in session.query(Job.job_id)
            .filter(Job.tenant_id == tenant_id)
            .yield_per(_PURGE_SCAN_BATCH_SIZE)
        ]
        try:
            quiesce_timeout = float(
                os.environ.get("TENANT_PURGE_QUIESCE_SECONDS", "30")
            )
        except (TypeError, ValueError):
            quiesce_timeout = 30.0
        quiesce_timeout = min(max(quiesce_timeout, 1.0), 300.0)
        get_manager().quiesce_jobs_for_tenant(
            tenant_id,
            quiesce_ids,
            timeout=quiesce_timeout,
        )
        _record_tenant_purge_event(
            tenant_id=tenant_id,
            purge_id=purge_id,
            event_type="tenant_purge_writers_quiesced",
            data={"job_count": len(quiesce_ids)},
        )
        session.expire_all()

        # Capture resource IDs before deleting their parent rows.  The IDs
        # also let purge remove legacy webhook-DLQ entries that predate the
        # durable tenant stamp.
        tenant_job_ids: set[str] = set()
        batch_rows = list(
            session.query(Batch)
            .filter(Batch.tenant_id == tenant_id)
            .yield_per(_PURGE_SCAN_BATCH_SIZE)
        )
        legacy_batches = (
            session.query(Batch)
            .filter(Batch.tenant_id.is_(None))
            .yield_per(_PURGE_SCAN_BATCH_SIZE)
        )
        legacy_batches_skipped = 0
        for candidate in legacy_batches:
            try:
                settings = candidate.settings or {}
            except Exception:
                legacy_batches_skipped += 1
                logger.warning(
                    "Skipping legacy batch %s during purge of tenant %s: "
                    "settings unreadable; row excluded from purge snapshot",
                    getattr(candidate, "batch_id", "?"),
                    tenant_id,
                )
                settings = {}
            if isinstance(settings, dict) and settings.get("tenant_id") == tenant_id:
                batch_rows.append(candidate)
        if legacy_batches_skipped:
            logger.warning(
                "Tenant %s purge skipped %d legacy batch row(s) with "
                "unreadable settings",
                tenant_id,
                legacy_batches_skipped,
            )
        tenant_batch_ids = {str(batch.batch_id) for batch in batch_rows}

        job_rows = (
            session.query(
                Job.job_id,
                Job.result_path,
                Job.source_file,
                Job.source_hash,
            )
            .filter(Job.tenant_id == tenant_id)
            .yield_per(_PURGE_SCAN_BATCH_SIZE)
        )
        fs_removed = 0
        for job_id, result_path, source_file, source_hash in job_rows:
            tenant_job_ids.add(str(job_id))
            if include_fs:
                fs_removed += _purge_fs_job_artifacts(
                    job_id=str(job_id),
                    result_path=result_path,
                    tenant_id=tenant_id,
                    config_mod=config,
                    source_file=source_file,
                    source_hash=source_hash,
                )

        _record_tenant_purge_event(
            tenant_id=tenant_id,
            purge_id=purge_id,
            event_type="tenant_purge_filesystem_deleted",
            data={"fs_dirs": fs_removed, "include_fs": include_fs},
        )

        from api.webhook_dlq import purge_dlq_entries_for_tenant

        dlq_deleted = purge_dlq_entries_for_tenant(
            tenant_id,
            resource_ids=tenant_job_ids | tenant_batch_ids,
        )

        review_ids = set(tenant_job_ids) | set(tenant_batch_ids)
        review_ids.update(f"translation-batch:{bid}" for bid in tenant_batch_ids)
        auxiliary_deleted = _purge_auxiliary_api_data(
            tenant_id,
            review_ids,
        )
        _record_tenant_purge_event(
            tenant_id=tenant_id,
            purge_id=purge_id,
            event_type="tenant_purge_auxiliary_deleted",
            data={"webhook_dlq_entries": dlq_deleted, **auxiliary_deleted},
        )

        _record_tenant_purge_event(
            tenant_id=tenant_id,
            purge_id=purge_id,
            event_type="tenant_purge_sql_delete_started",
            data={
                "job_ids": len(tenant_job_ids),
                "batch_ids": len(tenant_batch_ids),
            },
        )

        jobs_deleted = (
            session.query(Job)
            .filter(Job.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )
        usage_deleted = (
            session.query(UsageRecord)
            .filter(UsageRecord.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )
        keys_deleted = (
            session.query(TenantApiKey)
            .filter(TenantApiKey.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )
        rate_limits_deleted = (
            session.query(TenantRateLimit)
            .filter(TenantRateLimit.tenant_id == tenant_id)
            .delete(synchronize_session="fetch")
        )

        batches_deleted = 0
        for batch in batch_rows:
            session.delete(batch)
            batches_deleted += 1

        session.delete(tenant)
        session.commit()
        sql_committed = True

        deleted_counts = {
            "jobs": jobs_deleted,
            "batches": batches_deleted,
            "usage_records": usage_deleted,
            "api_keys": keys_deleted,
            "rate_limits": rate_limits_deleted,
            "webhook_dlq_entries": dlq_deleted,
            "fs_dirs": fs_removed,
            "legacy_batches_skipped": legacy_batches_skipped,
            **auxiliary_deleted,
        }
        _record_tenant_purge_event(
            tenant_id=tenant_id,
            purge_id=purge_id,
            event_type="tenant_purge_completed",
            data=deleted_counts,
        )

        logger.info(
            "Purged tenant %s: %d jobs, %d batches, %d usage records, "
            "%d API keys, %d rate limits, %d DLQ entries, %d fs dirs",
            tenant_id,
            jobs_deleted,
            batches_deleted,
            usage_deleted,
            keys_deleted,
            rate_limits_deleted,
            dlq_deleted,
            fs_removed,
        )
        logger.info(
            "Purged tenant %s auxiliary stores: %s",
            tenant_id,
            auxiliary_deleted,
        )
        return {
            "tenant_id": tenant_id,
            "deleted": {
                **deleted_counts,
            },
        }
    except Exception as exc:
        session.rollback()
        try:
            _record_tenant_purge_event(
                tenant_id=tenant_id,
                purge_id=purge_id,
                event_type="tenant_purge_failed",
                data={
                    "error": f"{type(exc).__name__}: {exc}",
                    "sql_committed": sql_committed,
                },
            )
        except Exception:
            logger.exception("Could not persist purge failure custody event")
        if fenced:
            # The admission fence is intentionally short-lived, but a failed
            # purge must not strand a tenant in ``deleting`` forever.  Publish
            # a distinct fail-closed state so an operator can retry the purge
            # without ever re-enabling admission on partially deleted data.
            try:
                with _admission_serialization() as acquire_db_lock:
                    acquire_db_lock(session)
                    failed_tenant = session.get(Tenant, tenant_id)
                    if failed_tenant and failed_tenant.status == "deleting":
                        failed_tenant.status = PURGE_FAILED_STATUS
                        failed_tenant.updated_at = _utcnow()
                        session.commit()
            except Exception:
                session.rollback()
                logger.exception(
                    "Could not persist purge recovery state for tenant %s",
                    tenant_id,
                )
        raise
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Cost bridge: map database-backed usage to cost_tracking.py schema
# ---------------------------------------------------------------------------


def build_cost_tracking_bridge(
    tenant_id: str,
    *,
    session: Optional[Session] = None,
) -> Optional[dict]:
    """Return a cost report in the same schema as cost_tracking.TenantUsage.estimated_cost().

    Reads from the database-backed UsageRecord (api.usage) and formats
    the result to match the standalone cost_tracking.py schema, providing
    a bridge between the admin API and the standalone cost tracker.

    Returns None if no usage record exists for the current period.
    """
    from api.usage import get_usage

    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        usage = get_usage(tenant_id, session=session)
        if not usage:
            return None

        # Import standalone cost rates for compatibility
        try:
            from cost_tracking import (
                COST_PER_API_CALL,
                COST_PER_GB_STORED,
                COST_PER_GPU_SECOND,
                COST_PER_PAGE,
            )
        except ImportError:
            # Fallback defaults matching cost_tracking.py
            COST_PER_PAGE = 0.01
            COST_PER_GPU_SECOND = 0.001
            COST_PER_GB_STORED = 0.05
            COST_PER_API_CALL = 0.0001

        pages = int(usage.pages_processed or 0)
        storage_bytes = int(usage.storage_bytes_used or 0)
        api_calls = int(usage.api_calls or 0)
        processing_seconds = float(usage.processing_seconds or 0.0)
        storage_gb = storage_bytes / (1024**3)

        page_cost = pages * COST_PER_PAGE
        gpu_cost = processing_seconds * COST_PER_GPU_SECOND
        storage_cost = storage_gb * COST_PER_GB_STORED
        api_cost = api_calls * COST_PER_API_CALL

        return {
            "tenant_id": tenant_id,
            "usage": {
                "tenant_id": tenant_id,
                "pages_processed": pages,
                "gpu_seconds": processing_seconds,
                "storage_bytes": storage_bytes,
                "api_calls": api_calls,
                "jobs_submitted": int(usage.jobs_submitted or 0),
                "jobs_completed": 0,
                "jobs_failed": 0,
                "first_activity": "",
                "last_activity": "",
            },
            "cost": {
                "page_cost": round(page_cost, 4),
                "gpu_cost": round(gpu_cost, 4),
                "storage_cost": round(storage_cost, 4),
                "api_cost": round(api_cost, 4),
                "total_cost": round(page_cost + gpu_cost + storage_cost + api_cost, 4),
                "currency": "USD",
            },
        }
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# API Key management
# ---------------------------------------------------------------------------


def validate_api_key_permissions(permissions: object) -> list[str]:
    """Validate and copy the closed tenant-key capability list."""
    from api.authz import VALID_TENANT_PERMISSIONS

    if not isinstance(permissions, list) or not permissions:
        raise ValueError("API key permissions must be a non-empty list")
    if len(permissions) > len(VALID_TENANT_PERMISSIONS):
        raise ValueError("API key permissions contain too many entries")
    if any(
        not isinstance(permission, str) or permission not in VALID_TENANT_PERMISSIONS
        for permission in permissions
    ):
        raise ValueError("API key permissions contain an unsupported capability")
    if len(set(permissions)) != len(permissions):
        raise ValueError("API key permissions must not contain duplicates")
    return list(permissions)


def create_api_key(
    tenant_id: str,
    *,
    name: Optional[str] = None,
    permissions: Optional[list[str]] = None,
    expires_at: Optional[datetime] = None,
    session: Optional[Session] = None,
) -> tuple[str, str]:
    """Create a new API key for a tenant.

    Returns (key_id, raw_api_key). The raw key is returned only once
    and is never stored — only its SHA-256 hash is persisted.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        with _admission_serialization() as acquire_db_lock:
            acquire_db_lock(session)
            tenant = session.get(Tenant, tenant_id)
            if not tenant or tenant.status != "active":
                raise ValueError(f"Tenant not found or inactive: {tenant_id}")

            raw_key = _generate_api_key()
            key_id = _generate_key_id()

            effective_permissions = (
                ["submit", "read"] if permissions is None else permissions
            )
            effective_permissions = validate_api_key_permissions(effective_permissions)

            record = TenantApiKey(
                key_id=key_id,
                tenant_id=tenant_id,
                api_key_hash=hash_api_key(raw_key),
                name=name,
                status="active",
                permissions=json.dumps(effective_permissions),
                created_at=_utcnow(),
                expires_at=expires_at,
            )
            session.add(record)
            session.commit()
            logger.info("Created API key %s for tenant %s", key_id, tenant_id)
            return key_id, raw_key
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def revoke_api_key(
    key_id: str,
    *,
    session: Optional[Session] = None,
) -> bool:
    """Revoke an API key. Returns True if key was found and revoked."""
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        record = session.get(TenantApiKey, key_id)
        if not record:
            return False
        record.status = "revoked"
        session.commit()
        logger.info("Revoked API key %s", key_id)
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def resolve_tenant_by_key(
    api_key_hash: str,
    *,
    session: Optional[Session] = None,
) -> Optional[tuple[Tenant, TenantApiKey]]:
    """Look up a tenant by the SHA-256 hash of the provided API key.

    Returns (Tenant, TenantApiKey) or None if no active match is found.
    Expired and revoked keys are excluded.
    """
    own_session = session is None
    if own_session:
        session = get_session_factory()()
    try:
        key_record = (
            session.query(TenantApiKey)
            .filter(
                TenantApiKey.api_key_hash == api_key_hash,
                TenantApiKey.status == "active",
            )
            .first()
        )
        if not key_record:
            return None

        # Check expiry
        now = _utcnow()
        if key_record.expires_at and key_record.expires_at < now:
            return None

        tenant = session.get(Tenant, key_record.tenant_id)
        if not tenant or tenant.status != "active":
            return None

        # Avoid a write on every request under sustained traffic.
        if (
            key_record.last_used_at is None
            or now - key_record.last_used_at >= timedelta(minutes=5)
        ):
            key_record.last_used_at = now
            session.commit()

        return tenant, key_record
    finally:
        if own_session:
            session.close()
