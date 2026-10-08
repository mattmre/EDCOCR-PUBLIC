"""Batch lifecycle manager -- coordinates multiple jobs as a single batch."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import BinaryIO, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from api import config
from api.database import Batch, Job, Tenant

logger = logging.getLogger(__name__)


def _safe_rejected_input_label(value: object) -> str:
    """Keep batch failure metadata from disclosing host filesystem paths."""
    raw = str(value or "").replace("\\", "/")
    label = raw.rsplit("/", 1)[-1]
    return (label or "<unnamed>")[:255]


def _batch_upload_budget_bytes() -> int:
    """Return the aggregate multipart budget for one API batch."""
    raw = getattr(config, "MAX_BATCH_UPLOAD_SIZE_MB", None)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raw = getattr(config, "MAX_UPLOAD_SIZE_MB", 5120)
    try:
        megabytes = max(1, int(raw))
    except (TypeError, ValueError):
        megabytes = 5120
    return megabytes * 1024 * 1024


def _measure_batch_uploads(files: list[tuple[str, bytes | BinaryIO]]) -> None:
    """Enforce a bounded aggregate upload budget without buffering streams."""
    budget = _batch_upload_budget_bytes()
    total = 0
    for filename, content in files:
        if isinstance(content, (bytes, bytearray)):
            size = len(content)
        else:
            try:
                current = content.tell()
                content.seek(0, 2)
                size = content.tell()
                content.seek(current)
            except (AttributeError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Batch upload stream for {_safe_rejected_input_label(filename)} "
                    "must support bounded size inspection"
                ) from exc
        if size < 0 or total + size > budget:
            raise ValueError(
                "Aggregate batch upload exceeds maximum of "
                f"{budget // (1024 * 1024)} MB"
            )
        total += size


# Terminal statuses for individual jobs
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_BATCH_TERMINAL_STATUSES = _TERMINAL_STATUSES | {"partial_failure"}


def _serialize_batch_retries(method):
    @wraps(method)
    def serialized(self, *args, **kwargs):
        from api.job_manager import _batch_retry_serialization_lock

        # Process death releases the OS lock, allowing recovery of abandoned claims.
        with _batch_retry_serialization_lock():
            return method(self, *args, **kwargs)

    return serialized


class BatchManager:
    """Manages batch submission, tracking, and lifecycle."""

    def __init__(self, db_session_factory, job_manager=None):
        self._session_factory = db_session_factory
        self._job_manager = job_manager

    def _publish_batch_event(self, batch, event_type: str) -> None:
        """Persist a batch lifecycle event."""
        from api.events import publish_batch_event

        publish_batch_event(batch, event_type)

    def recover_abandoned_retry_claims(self, batch_id: str) -> int:
        """Release claims left by a retry process that exited unexpectedly.

        The caller must hold ``_batch_retry_serialization_lock()`` so this
        only runs after any live retry request has left its critical section.
        Admitted replacements are retained in the batch; originals without
        replacements remain retryable.
        """
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if batch is None:
                logger.error(
                    "Cannot recover retry claims for missing batch %s", batch_id
                )
                return 0
            batch_tenant_id = self._batch_tenant_id(batch)
            claimed = (
                session.query(Job)
                .filter(
                    Job.batch_id == batch_id,
                    Job.retry_claimed == 1,
                )
                .all()
            )
            recovered = 0
            for original in claimed:
                if original.tenant_id != batch_tenant_id:
                    logger.error(
                        "Cannot recover retry claim for job %s in batch %s: "
                        "original tenant does not match batch tenant",
                        original.job_id,
                        batch_id,
                    )
                    continue
                replacement = (
                    session.get(Job, original.retried_to)
                    if original.retried_to
                    else None
                )
                if replacement is not None:
                    replacement_settings = (
                        replacement.settings
                        if isinstance(replacement.settings, dict)
                        else {}
                    )
                    if (
                        replacement.job_id == original.job_id
                        or replacement.retried_to == original.job_id
                        or replacement.retried_to is not None
                        or replacement_settings.get("retried_to")
                        or replacement_settings.get("superseded")
                    ):
                        logger.error(
                            "Cannot recover retry claim for job %s in batch %s: "
                            "replacement link is self-referential or superseded",
                            original.job_id,
                            batch_id,
                        )
                        continue
                    if replacement.tenant_id != batch_tenant_id:
                        logger.error(
                            "Cannot recover retry claim for job %s in batch %s: "
                            "replacement %s has a tenant mismatch",
                            original.job_id,
                            batch_id,
                            replacement.job_id,
                        )
                        continue
                    if replacement.batch_id not in (None, batch_id):
                        # Do not release a claim when its replacement is
                        # attached to a different batch. That would allow this
                        # parent to finalize around a live child and hide an
                        # inconsistent cross-batch retry link.
                        logger.error(
                            "Cannot recover retry claim for job %s in batch %s: "
                            "replacement %s belongs to batch %s",
                            original.job_id,
                            batch_id,
                            replacement.job_id,
                            replacement.batch_id,
                        )
                        continue
                    if replacement.batch_id is None:
                        # An individual retry can race a claimed batch retry.
                        # Adopt its already-created replacement before the
                        # parent claim is released, so completion accounting
                        # follows the live retry instead of the failed source.
                        replacement.batch_id = batch_id
                    original.batch_id = None
                    settings = dict(original.settings or {})
                    settings.update(
                        {
                            "retried_to": replacement.job_id,
                            "superseded": True,
                            "previous_batch_id": batch_id,
                        }
                    )
                    original.settings = settings
                else:
                    # A crash before replacement insertion can leave a stale
                    # retry pointer in partially migrated/legacy rows. Make
                    # the original eligible for a later retry again.
                    original.retried_to = None
                    settings = dict(original.settings or {})
                    settings.pop("retried_to", None)
                    settings.pop("superseded", None)
                    settings.pop("previous_batch_id", None)
                    original.settings = settings
                original.retry_claimed = 0
                recovered += 1
            if recovered:
                batch.total_jobs = (
                    session.query(Job).filter(Job.batch_id == batch_id).count()
                )
                session.commit()
            return recovered
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        *,
        files: Optional[list[tuple[str, bytes | BinaryIO]]] = None,
        source_paths: Optional[list[str]] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> tuple[Batch, list[Job]]:
        """Create a batch and submit child jobs for each file/path.

        Parameters
        ----------
        files : list of (filename, content) tuples from file uploads
        source_paths : list of server-side file paths
        priority : job priority level
        enable_docintel : enable document intelligence
        docintel_mode : document intelligence mode
        webhook_url : batch-level webhook URL
        webhook_secret : batch-level webhook secret
        tenant_id : authenticated tenant scope (stored in settings for isolation)

        Returns
        -------
        tuple of (Batch, list[Job])

        Raises
        ------
        ValueError
            If no files or paths provided, or batch size exceeds limit.
        """
        if skip_ocr:
            # Batch children use the same standalone API pipeline as single
            # jobs; refuse before creating the parent or any child.
            raise ValueError("skip_ocr is not supported by the standalone API pipeline")
        file_list = files or []
        path_list = source_paths or []
        total = len(file_list) + len(path_list)

        if total == 0:
            raise ValueError("At least one file or source_path is required.")

        max_batch_size = getattr(config, "MAX_BATCH_SIZE", 50)
        try:
            max_batch_size = int(max_batch_size)
        except (TypeError, ValueError):
            max_batch_size = 50
        max_batch_size = max(1, max_batch_size)
        if total > max_batch_size:
            raise ValueError(f"Batch size {total} exceeds maximum of {max_batch_size}.")

        _measure_batch_uploads(file_list)

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        settings = {
            "enable_docintel": enable_docintel,
            "docintel_mode": docintel_mode,
        }
        if processing_timeout_minutes is not None:
            settings["processing_timeout_minutes"] = processing_timeout_minutes
        # Keep settings stamp for backward-compatible readers; durable pin is column.
        if tenant_id:
            settings["tenant_id"] = tenant_id

        # Encrypt webhook secret before storing at rest (SEC-001)
        encrypted_secret = None
        if webhook_secret:
            from api.config import encrypt_webhook_secret

            encrypted_secret = encrypt_webhook_secret(webhook_secret)

        session: Session = self._session_factory()
        try:
            from api.tenant_manager import _admission_serialization

            with _admission_serialization() as acquire_db_lock:
                acquire_db_lock(session)
                if tenant_id:
                    tenant = session.get(Tenant, tenant_id)
                    if tenant is None or tenant.status != "active":
                        raise ValueError("Tenant not found or inactive")
                batch = Batch(
                    batch_id=batch_id,
                    status="submitted",
                    total_jobs=total,
                    priority=priority,
                    webhook_url=webhook_url,
                    webhook_secret=encrypted_secret,
                    tenant_id=tenant_id if tenant_id else None,
                )
                batch.settings = settings
                session.add(batch)
                session.commit()
                session.refresh(batch)
        finally:
            session.close()

        # Submit individual jobs via the JobManager
        child_jobs = []
        rejected_inputs: list[str] = []

        for filename, content in file_list:
            try:
                upload_kwargs = {
                    "upload_filename": filename,
                    "priority": priority,
                    "enable_docintel": enable_docintel,
                    "docintel_mode": docintel_mode,
                    "processing_timeout_minutes": processing_timeout_minutes,
                    "tenant_id": tenant_id,
                    "batch_id": batch_id,
                }
                if isinstance(content, (bytes, bytearray)):
                    upload_kwargs["upload_content"] = content
                else:
                    upload_kwargs["upload_file"] = content
                job = self._job_manager.submit(**upload_kwargs)
                child_jobs.append(job)
            except Exception:
                logger.exception(
                    "Failed to submit file %s in batch %s",
                    _safe_rejected_input_label(filename),
                    batch_id,
                )
                rejected_inputs.append(_safe_rejected_input_label(filename))

        for path in path_list:
            try:
                job = self._job_manager.submit(
                    source_path=path,
                    priority=priority,
                    enable_docintel=enable_docintel,
                    docintel_mode=docintel_mode,
                    skip_ocr=skip_ocr,
                    processing_timeout_minutes=processing_timeout_minutes,
                    tenant_id=tenant_id,
                    batch_id=batch_id,
                )
                child_jobs.append(job)
            except Exception:
                logger.exception(
                    "Failed to submit path %s in batch %s",
                    _safe_rejected_input_label(path),
                    batch_id,
                )
                rejected_inputs.append(_safe_rejected_input_label(path))

        # Update batch with actual submitted count
        session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if batch:
                batch.total_jobs = len(child_jobs)
                try:
                    settings = batch.settings
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.warning(
                        "Resetting malformed settings for batch %s before update",
                        batch_id,
                    )
                    settings = {}
                if not isinstance(settings, dict):
                    settings = {}
                settings["rejected_inputs"] = rejected_inputs
                batch.settings = settings
                if len(child_jobs) == 0:
                    batch.status = "failed"
                session.commit()
                session.refresh(batch)
                event_type = (
                    "batch.failed" if batch.status == "failed" else "batch.submitted"
                )
                self._publish_batch_event(batch, event_type)
        finally:
            session.close()

        self.check_batch_completion(batch_id)
        batch = self.get_batch(batch_id)
        return batch, child_jobs

    def _set_job_batch_id(self, job_id: str, batch_id: str) -> None:
        """Associate an unbatched, same-tenant job without reassigning it."""
        session: Session = self._session_factory()
        try:
            job = (
                session.query(Job)
                .filter(Job.job_id == job_id)
                .with_for_update()
                .one_or_none()
            )
            batch = session.get(Batch, batch_id)
            if job is None or batch is None:
                raise ValueError("Retry job or target batch no longer exists")
            if job.batch_id not in (None, batch_id):
                raise ValueError("Retry job already belongs to another batch")
            if job.tenant_id != self._batch_tenant_id(batch):
                raise ValueError("Retry job tenant does not match target batch")
            job.batch_id = batch_id
            session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _batch_tenant_id(self, batch: Batch) -> Optional[str]:
        """Durable tenant scope: column first, then legacy settings stamp.

        Returns None only for unstamped legacy batches (pre-RC-03 residual fix).
        """
        col = getattr(batch, "tenant_id", None)
        if isinstance(col, str) and col.strip():
            return col.strip()
        try:
            settings = batch.settings or {}
        except Exception:
            settings = {}
        tid = settings.get("tenant_id") if isinstance(settings, dict) else None
        return tid if isinstance(tid, str) and tid else None

    def _has_foreign_batch_member(self, session: Session, batch_id: str, batch: Batch) -> bool:
        """Fail closed if a batch contains a child outside its tenant scope.

        Legacy unstamped batches may contain only unstamped children. A child
        with a tenant stamp cannot be safely aggregated into such a batch.
        """
        batch_tenant_id = self._batch_tenant_id(batch)
        membership = session.query(Job.job_id).filter(Job.batch_id == batch_id)
        if batch_tenant_id is None:
            membership = membership.filter(Job.tenant_id.is_not(None))
        else:
            membership = membership.filter(
                or_(Job.tenant_id.is_(None), Job.tenant_id != batch_tenant_id)
            )
        return membership.first() is not None

    def _batch_owned_by(self, batch: Batch, tenant_id: Optional[str]) -> bool:
        """RC-03: scoped access requires durable stamp match; no any-child admit.

        Unstamped legacy batches are invisible under a tenant-scoped call.
        Platform-admin / single-tenant callers pass tenant_id=None.
        """
        if tenant_id is None:
            return True
        owned = self._batch_tenant_id(batch)
        if owned is None:
            return False
        return owned == tenant_id

    def get_batch(
        self, batch_id: str, tenant_id: Optional[str] = None
    ) -> Optional[Batch]:
        """Return the batch record, or None if not found / wrong tenant."""
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if batch is None:
                return None
            if not self._batch_owned_by(batch, tenant_id):
                return None
            return batch
        finally:
            session.close()

    def get_batch_jobs(
        self, batch_id: str, tenant_id: Optional[str] = None
    ) -> list[Job]:
        """Return jobs belonging to a batch, optionally tenant-scoped.

        A batch lookup and its child lookup are separate database queries.
        Keep the tenant predicate on the child query as defense in depth so
        a malformed/legacy mixed-tenant batch cannot turn a correctly scoped
        batch response into a cross-tenant disclosure.
        """
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.batch_id == batch_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            return query.order_by(Job.created_at).all()
        finally:
            session.close()

    def list_batches(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        tenant_id: Optional[str] = None,
        # Backward-compatible aliases (deprecated)
        page: Optional[int] = None,
        per_page: Optional[int] = None,
    ) -> tuple[list[Batch], int]:
        """Return a paginated list of batches with optional status filter.

        Accepts canonical ``limit``/``offset`` or legacy ``page``/``per_page``.

        Returns
        -------
        tuple of (list[Batch], total_count)
        """
        # Keep internal callers under the same bounds as the HTTP dependency.
        from api.deps import normalize_pagination

        limit, offset = normalize_pagination(limit, offset, page, per_page)

        session: Session = self._session_factory()
        try:
            q = session.query(Batch)
            if status:
                q = q.filter(Batch.status == status)
            if tenant_id is not None:
                # RC-03: query the durable owner column before pagination. The
                # previous implementation materialized every batch for the
                # tenant and only then sliced it in Python.
                q = q.filter(Batch.tenant_id == tenant_id)
                total = q.count()
                batches = (
                    q.order_by(Batch.created_at.desc(), Batch.batch_id.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                # Keep a final object-level check for malformed rows and for
                # test/dialect adapters that do not enforce SQL predicates.
                visible_batches = [
                    b for b in batches if self._batch_owned_by(b, tenant_id)
                ]
                # A real SQLAlchemy query already counted the tenant-scoped
                # rows.  If an adapter ignored that predicate, however, do
                # not return a foreign-tenant count to the caller.  Adjust
                # only for rows removed from this bounded page; never scan
                # the complete table just to repair a defensive count.
                total = max(0, total - (len(batches) - len(visible_batches)))
                batches = visible_batches
                return batches, total
            total = q.count()
            batches = (
                q.order_by(Batch.created_at.desc(), Batch.batch_id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return batches, total
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_batch(
        self, batch_id: str, tenant_id: Optional[str] = None
    ) -> Optional[Batch]:
        """Cancel all non-terminal child jobs and update batch status.

        When *tenant_id* is set, returns None (not found) if the batch is not
        owned by that tenant — same isolation as get_batch.
        """
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return None
            if not self._batch_owned_by(batch, tenant_id):
                return None

            if batch.status in _BATCH_TERMINAL_STATUSES:
                return batch

            jobs_q = session.query(Job).filter(Job.batch_id == batch_id)
            # R4-RC-03: never mutate foreign-tenant children under a scoped call.
            if tenant_id is not None:
                jobs_q = jobs_q.filter(Job.tenant_id == tenant_id)
            jobs = jobs_q.all()

            cancelled_count = 0
            for job in jobs:
                if job.status not in _TERMINAL_STATUSES:
                    if self._job_manager:
                        # Always pin cancel_job to authorizing tenant scope
                        # (None only for platform / single-tenant unscoped).
                        result = self._job_manager.cancel_job(
                            job.job_id, tenant_id=tenant_id
                        )
                        if result is not None:
                            cancelled_count += 1

            # Refresh batch counters from DB
            session.expire(batch)
            self._update_batch_counters(batch_id)

            batch = session.get(Batch, batch_id)
            if batch:
                # Batch status is global to the batch, not to the caller's
                # child subset.  A malformed mixed-tenant batch must not be
                # marked cancelled while an out-of-scope child is still
                # running.
                remaining = session.query(Job).filter(
                    Job.batch_id == batch_id,
                    Job.status.notin_(list(_TERMINAL_STATUSES)),
                )
                if remaining.count() == 0:
                    completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    transitioned = (
                        session.query(Batch)
                        .filter(
                            Batch.batch_id == batch_id,
                            Batch.status.notin_(list(_BATCH_TERMINAL_STATUSES)),
                        )
                        .update(
                            {"status": "cancelled", "completed_at": completed_at},
                            synchronize_session=False,
                        )
                    )
                    session.commit()
                    if transitioned:
                        batch = session.get(Batch, batch_id)
                        self._publish_batch_event(batch, "batch.cancelled")

            return batch
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    @_serialize_batch_retries
    def retry_batch(
        self, batch_id: str, tenant_id: Optional[str] = None
    ) -> Optional[tuple[Batch, list[Job]]]:
        """Retry all failed/cancelled child jobs in a batch.

        When *tenant_id* is set, returns None if the batch is not owned by
        that tenant. Each child retry is tenant-pinned (R4-RC-03). Successful
        originals are detached from the batch so re-retry cannot unbounded-clone.

        Returns
        -------
        tuple of (Batch, list of new Job records) or None if batch not found
        """
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return None
            if not self._batch_owned_by(batch, tenant_id):
                return None

            batch_tenant_id = self._batch_tenant_id(batch)
            batch_members = (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .all()
            )
            if any(job.tenant_id != batch_tenant_id for job in batch_members):
                logger.error(
                    "Refusing retry of mixed-tenant batch %s", batch_id
                )
                raise ValueError("Batch contains jobs outside its tenant membership")

            # No retry or reconciler can still own a claim while this shared
            # fence is held. Recover both durable links and legacy settings
            # before selecting new work; a raw claim reset could leave a
            # stale settings-only superseded marker that is never retryable.
            session.rollback()
            self.recover_abandoned_retry_claims(batch_id)
            session.expire_all()
            batch = session.get(Batch, batch_id)
            if batch is None or not self._batch_owned_by(batch, tenant_id):
                return None

            failed_q = session.query(Job).filter(
                Job.batch_id == batch_id,
                Job.status.in_(["failed", "cancelled"]),
            )
            if tenant_id is not None:
                failed_q = failed_q.filter(Job.tenant_id == tenant_id)
            failed_jobs = failed_q.all()

            # Skip already-superseded originals (settings.retried_to set).
            retryable = []
            for j in failed_jobs:
                settings = j.settings if isinstance(j.settings, dict) else {}
                if (
                    j.retried_to
                    or settings.get("retried_to")
                    or settings.get("superseded")
                ):
                    continue
                retryable.append(j)

            if not retryable:
                raise ValueError("No failed or cancelled jobs to retry in this batch.")

            # Move the parent out of a terminal state before children are
            # cloned. Completion workers will therefore not finalize the
            # batch during the association window.
            session.query(Batch).filter(
                Batch.batch_id == batch_id,
                Batch.status.in_(
                    [
                        "submitted",
                        "processing",
                        "failed",
                        "partial_failure",
                        "cancelled",
                    ]
                ),
            ).update(
                {"status": "processing", "completed_at": None},
                synchronize_session=False,
            )
            # Claim the whole request before starting any replacement.
            # Otherwise an immediately failed first replacement can finalize
            # the parent while later failed originals are still unclaimed.
            retry_ids = [job.job_id for job in retryable]
            session.query(Job).filter(
                Job.job_id.in_(retry_ids),
                Job.batch_id == batch_id,
                Job.status.in_(["failed", "cancelled"]),
                or_(Job.retry_claimed == 0, Job.retry_claimed.is_(None)),
            ).update({"retry_claimed": 1}, synchronize_session=False)
            session.commit()
            session.expire_all()
            batch = session.get(Batch, batch_id)

            new_jobs = []
            for old_job in retryable:
                if old_job.retry_claimed != 1 or old_job.batch_id != batch_id:
                    continue
                preserve_claim = False
                try:
                    new_job = self._job_manager.retry_job(
                        old_job.job_id,
                        tenant_id=tenant_id,
                        batch_id=batch_id,
                    )
                    if new_job is None:
                        raise ValueError("Retry did not create a replacement job")
                    if new_job:
                        if new_job.tenant_id != old_job.tenant_id:
                            preserve_claim = True
                            raise ValueError(
                                "Retry replacement tenant does not match source"
                            )
                        # Older/mocked JobManagers may not support the atomic
                        # batch argument. Production JobManager associates the
                        # replacement in its initial INSERT transaction.
                        replacement_batch_id = getattr(new_job, "batch_id", None)
                        if replacement_batch_id not in (None, batch_id):
                            preserve_claim = True
                            raise ValueError(
                                "Retry replacement belongs to another batch"
                            )
                        if replacement_batch_id is None:
                            preserve_claim = True
                            self._set_job_batch_id(new_job.job_id, batch_id)
                            new_job.batch_id = batch_id
                        # Detach original from batch + mark superseded so a
                        # second POST .../retry cannot re-clone the same row.
                        original = session.get(Job, old_job.job_id)
                        if original is not None:
                            settings = dict(original.settings or {})
                            settings["retried_to"] = new_job.job_id
                            settings["superseded"] = True
                            settings["previous_batch_id"] = batch_id
                            original.settings = settings
                            original.batch_id = None
                            original.retry_claimed = 0
                            original.retried_to = new_job.job_id
                            session.commit()
                        new_jobs.append(new_job)
                except Exception:
                    session.rollback()
                    if not preserve_claim:
                        session.query(Job).filter(
                            Job.job_id == old_job.job_id,
                            Job.batch_id == batch_id,
                            Job.retry_claimed == 1,
                            Job.retried_to.is_(None),
                        ).update({"retry_claimed": 0}, synchronize_session=False)
                        session.commit()
                    logger.exception(
                        "Failed to retry job %s in batch %s",
                        old_job.job_id,
                        batch_id,
                    )

            if not new_jobs:
                self.check_batch_completion(batch_id)
                raise ValueError("No jobs could be retried successfully in this batch.")

            # Update batch counters only after at least one productive retry.
            all_jobs = session.query(Job).filter(Job.batch_id == batch_id).all()
            session.refresh(batch)
            batch.total_jobs = len(all_jobs)
            # A replacement may already have finalized this parent while
            # retry_job returned. Preserve that terminal transition.
            session.commit()
            session.refresh(batch)
            if batch.status == "processing":
                self._publish_batch_event(batch, "batch.processing")
            self.check_batch_completion(batch_id)
            session.refresh(batch)

            return batch, new_jobs
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Batch completion detection
    # ------------------------------------------------------------------

    def check_batch_completion(self, batch_id: str) -> None:
        """Check if all jobs in a batch are terminal and update status.

        Called after each child job completes. If all children are in terminal
        states, derives the batch status and fires the batch-level webhook.
        """
        session: Session = self._session_factory()
        try:
            # Serialize completion decisions per batch.  The conditional
            # update below remains the duplicate-publication guard on SQLite,
            # where SELECT FOR UPDATE is not enforced.
            batch = (
                session.query(Batch)
                .filter(Batch.batch_id == batch_id)
                .with_for_update()
                .one_or_none()
            )
            if not batch:
                return
            if self._has_foreign_batch_member(session, batch_id, batch):
                logger.error(
                    "Refusing completion aggregation for mixed-tenant batch %s",
                    batch_id,
                )
                return
            if batch.status in _BATCH_TERMINAL_STATUSES:
                return

            status_counts = dict(
                session.query(Job.status, func.count(Job.job_id))
                .filter(Job.batch_id == batch_id)
                .group_by(Job.status)
                .all()
            )
            total = sum(status_counts.values())
            if not total:
                return

            # Aggregate in SQL rather than materializing every child row.
            completed = int(status_counts.get("completed", 0))
            failed = int(status_counts.get("failed", 0))
            cancelled = int(status_counts.get("cancelled", 0))
            terminal = completed + failed + cancelled
            # Admission has not finished while expected inputs or claimed
            # retry replacements are absent. A fast first child must not
            # finalize its parent before later children are inserted.
            retry_pending = (
                session.query(Job)
                .filter(Job.batch_id == batch_id, Job.retry_claimed == 1)
                .first()
                is not None
            )
            all_terminal = terminal == total == batch.total_jobs and not retry_pending

            batch.jobs_completed = completed
            batch.jobs_failed = failed
            batch.jobs_cancelled = cancelled

            if all_terminal:
                completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                if completed == total:
                    next_status = "completed"
                elif failed == total:
                    next_status = "failed"
                elif cancelled == total:
                    next_status = "cancelled"
                else:
                    next_status = "partial_failure"
                processing_time = None
                if batch.created_at:
                    processing_time = (completed_at - batch.created_at).total_seconds()
                transitioned = (
                    session.query(Batch)
                    .filter(
                        Batch.batch_id == batch_id,
                        Batch.status.notin_(list(_BATCH_TERMINAL_STATUSES)),
                    )
                    .update(
                        {
                            "status": next_status,
                            "completed_at": completed_at,
                            "processing_time": processing_time,
                            "jobs_completed": completed,
                            "jobs_failed": failed,
                            "jobs_cancelled": cancelled,
                        },
                        synchronize_session=False,
                    )
                )
                session.commit()
                if transitioned:
                    batch = session.get(Batch, batch_id)
                    self._publish_batch_event(batch, f"batch.{next_status}")
                    if batch.webhook_url:
                        self._fire_batch_webhook(batch_id)
                return

            session.commit()

            # Fire webhook if batch is terminal
            # (Terminal transitions return above so only one caller can emit.)

        except Exception:
            logger.exception("Error checking batch completion for %s", batch_id)
            raise
        finally:
            session.close()

    def _update_batch_counters(self, batch_id: str) -> None:
        """Refresh batch counters from child job statuses."""
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return

            status_counts = dict(
                session.query(Job.status, func.count(Job.job_id))
                .filter(Job.batch_id == batch_id)
                .group_by(Job.status)
                .all()
            )
            batch.jobs_completed = int(status_counts.get("completed", 0))
            batch.jobs_failed = int(status_counts.get("failed", 0))
            batch.jobs_cancelled = int(status_counts.get("cancelled", 0))
            session.commit()
        finally:
            session.close()

    def _fire_batch_webhook(self, batch_id: str) -> None:
        """Start webhook delivery for a completed batch."""
        from api.webhooks import start_batch_webhook_delivery

        start_batch_webhook_delivery(
            batch_id,
            self._session_factory,
            webhook_timeout=config.WEBHOOK_TIMEOUT,
            webhook_max_retries=config.WEBHOOK_MAX_RETRIES,
            webhook_secret_default=config.WEBHOOK_SECRET,
        )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def derive_batch_status(jobs: list) -> str:
    """Derive batch status from child job statuses.

    Parameters
    ----------
    jobs : list of Job objects with a ``status`` attribute

    Returns
    -------
    str
        One of: submitted, processing, completed, partial_failure,
        failed, cancelled.
    """
    if not jobs:
        return "submitted"

    statuses = [j.status for j in jobs]
    total = len(statuses)

    completed = statuses.count("completed")
    failed = statuses.count("failed")
    cancelled = statuses.count("cancelled")
    terminal = completed + failed + cancelled

    # Not all done yet
    if terminal < total:
        if any(s == "processing" for s in statuses):
            return "processing"
        return "submitted"

    # All terminal
    if completed == total:
        return "completed"
    if failed == total:
        return "failed"
    if cancelled == total:
        return "cancelled"
    # Mix of terminal states
    return "partial_failure"
