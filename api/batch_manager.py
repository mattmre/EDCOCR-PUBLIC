"""Batch lifecycle manager -- coordinates multiple jobs as a single batch."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from api import config
from api.database import Batch, Job

logger = logging.getLogger(__name__)

# Terminal statuses for individual jobs
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class BatchManager:
    """Manages batch submission, tracking, and lifecycle."""

    def __init__(self, db_session_factory, job_manager=None):
        self._session_factory = db_session_factory
        self._job_manager = job_manager

    def _publish_batch_event(self, batch, event_type: str) -> None:
        """Persist a batch lifecycle event."""
        from api.events import publish_batch_event

        publish_batch_event(batch, event_type)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        *,
        files: Optional[list[tuple[str, bytes]]] = None,
        source_paths: Optional[list[str]] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
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

        Returns
        -------
        tuple of (Batch, list[Job])

        Raises
        ------
        ValueError
            If no files or paths provided, or batch size exceeds limit.
        """
        file_list = files or []
        path_list = source_paths or []
        total = len(file_list) + len(path_list)

        if total == 0:
            raise ValueError("At least one file or source_path is required.")

        max_batch_size = getattr(config, "MAX_BATCH_SIZE", 50)
        if total > max_batch_size:
            raise ValueError(
                f"Batch size {total} exceeds maximum of {max_batch_size}."
            )

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        settings = {
            "enable_docintel": enable_docintel,
            "docintel_mode": docintel_mode,
        }
        if processing_timeout_minutes is not None:
            settings["processing_timeout_minutes"] = processing_timeout_minutes

        # Encrypt webhook secret before storing at rest (SEC-001)
        encrypted_secret = None
        if webhook_secret:
            from api.config import encrypt_webhook_secret

            encrypted_secret = encrypt_webhook_secret(webhook_secret)

        session: Session = self._session_factory()
        try:
            batch = Batch(
                batch_id=batch_id,
                status="submitted",
                total_jobs=total,
                priority=priority,
                webhook_url=webhook_url,
                webhook_secret=encrypted_secret,
            )
            batch.settings = settings
            session.add(batch)
            session.commit()
            session.refresh(batch)
        finally:
            session.close()

        # Submit individual jobs via the JobManager
        child_jobs = []

        for filename, content in file_list:
            try:
                job = self._job_manager.submit(
                    upload_filename=filename,
                    upload_content=content,
                    priority=priority,
                    enable_docintel=enable_docintel,
                    docintel_mode=docintel_mode,
                    processing_timeout_minutes=processing_timeout_minutes,
                )
                self._set_job_batch_id(job.job_id, batch_id)
                child_jobs.append(job)
            except Exception:
                logger.exception(
                    "Failed to submit file %s in batch %s", filename, batch_id
                )

        for path in path_list:
            try:
                job = self._job_manager.submit(
                    source_path=path,
                    priority=priority,
                    enable_docintel=enable_docintel,
                    docintel_mode=docintel_mode,
                    skip_ocr=skip_ocr,
                    processing_timeout_minutes=processing_timeout_minutes,
                )
                self._set_job_batch_id(job.job_id, batch_id)
                child_jobs.append(job)
            except Exception:
                logger.exception(
                    "Failed to submit path %s in batch %s", path, batch_id
                )

        # Update batch with actual submitted count
        session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if batch:
                batch.total_jobs = len(child_jobs)
                if len(child_jobs) == 0:
                    batch.status = "failed"
                session.commit()
                session.refresh(batch)
                event_type = "batch.failed" if batch.status == "failed" else "batch.submitted"
                self._publish_batch_event(batch, event_type)
        finally:
            session.close()

        return batch, child_jobs

    def _set_job_batch_id(self, job_id: str, batch_id: str) -> None:
        """Associate a job with a batch."""
        session: Session = self._session_factory()
        try:
            job = session.get(Job, job_id)
            if job:
                job.batch_id = batch_id
                session.commit()
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """Return the batch record, or None if not found."""
        session: Session = self._session_factory()
        try:
            return session.get(Batch, batch_id)
        finally:
            session.close()

    def get_batch_jobs(self, batch_id: str) -> list[Job]:
        """Return all jobs belonging to a batch."""
        session: Session = self._session_factory()
        try:
            return (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .order_by(Job.created_at)
                .all()
            )
        finally:
            session.close()

    def list_batches(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
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
        # Legacy callers may still pass page/per_page -- normalize to limit/offset
        if page is not None and per_page is not None:
            limit = per_page
            offset = (page - 1) * per_page
        elif page is not None:
            offset = (page - 1) * limit
        elif per_page is not None:
            limit = per_page

        session: Session = self._session_factory()
        try:
            q = session.query(Batch)
            if status:
                q = q.filter(Batch.status == status)
            total = q.count()
            batches = (
                q.order_by(Batch.created_at.desc())
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

    def cancel_batch(self, batch_id: str) -> Optional[Batch]:
        """Cancel all non-terminal child jobs and update batch status."""
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return None

            if batch.status in ("completed", "failed", "cancelled"):
                return batch

            jobs = (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .all()
            )

            cancelled_count = 0
            for job in jobs:
                if job.status not in _TERMINAL_STATUSES:
                    if self._job_manager:
                        self._job_manager.cancel_job(job.job_id)
                    cancelled_count += 1

            # Refresh batch counters from DB
            session.expire(batch)
            self._update_batch_counters(batch_id)

            batch = session.get(Batch, batch_id)
            if batch:
                batch.status = "cancelled"
                batch.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.commit()
                session.refresh(batch)
                self._publish_batch_event(batch, "batch.cancelled")

            return batch
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def retry_batch(self, batch_id: str) -> Optional[tuple[Batch, list[Job]]]:
        """Retry all failed/cancelled child jobs in a batch.

        Returns
        -------
        tuple of (Batch, list of new Job records) or None if batch not found
        """
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return None

            failed_jobs = (
                session.query(Job)
                .filter(
                    Job.batch_id == batch_id,
                    Job.status.in_(["failed", "cancelled"]),
                )
                .all()
            )

            if not failed_jobs:
                raise ValueError("No failed or cancelled jobs to retry in this batch.")

            new_jobs = []
            for old_job in failed_jobs:
                try:
                    new_job = self._job_manager.retry_job(old_job.job_id)
                    if new_job:
                        self._set_job_batch_id(new_job.job_id, batch_id)
                        new_jobs.append(new_job)
                except Exception:
                    logger.exception(
                        "Failed to retry job %s in batch %s",
                        old_job.job_id,
                        batch_id,
                    )

            # Update batch counters
            all_jobs = (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .all()
            )
            batch.total_jobs = len(all_jobs)
            batch.status = "processing"
            batch.completed_at = None
            session.commit()
            session.refresh(batch)
            self._publish_batch_event(batch, "batch.processing")

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
            batch = session.get(Batch, batch_id)
            if not batch:
                return
            if batch.status in ("completed", "failed", "cancelled"):
                return

            jobs = (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .all()
            )

            if not jobs:
                return

            # Update counters
            completed = sum(1 for j in jobs if j.status == "completed")
            failed = sum(1 for j in jobs if j.status == "failed")
            cancelled = sum(1 for j in jobs if j.status == "cancelled")
            all_terminal = all(j.status in _TERMINAL_STATUSES for j in jobs)

            batch.jobs_completed = completed
            batch.jobs_failed = failed
            batch.jobs_cancelled = cancelled

            if all_terminal:
                batch.status = derive_batch_status(jobs)
                batch.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                if batch.created_at:
                    delta = batch.completed_at - batch.created_at
                    batch.processing_time = delta.total_seconds()

            session.commit()
            if all_terminal:
                session.refresh(batch)
                self._publish_batch_event(batch, f"batch.{batch.status}")

            # Fire webhook if batch is terminal
            if all_terminal and batch.webhook_url:
                self._fire_batch_webhook(batch_id)

        except Exception:
            logger.exception(
                "Error checking batch completion for %s", batch_id
            )
        finally:
            session.close()

    def _update_batch_counters(self, batch_id: str) -> None:
        """Refresh batch counters from child job statuses."""
        session: Session = self._session_factory()
        try:
            batch = session.get(Batch, batch_id)
            if not batch:
                return

            jobs = (
                session.query(Job)
                .filter(Job.batch_id == batch_id)
                .all()
            )

            batch.jobs_completed = sum(1 for j in jobs if j.status == "completed")
            batch.jobs_failed = sum(1 for j in jobs if j.status == "failed")
            batch.jobs_cancelled = sum(1 for j in jobs if j.status == "cancelled")
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
