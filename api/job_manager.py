"""Job lifecycle manager — subprocess-based pipeline execution."""

from __future__ import annotations

import heapq
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from api import config
from api.database import Job, Tenant
from api.job_log_writer import write_job_log
from api.path_safety import validate_source_path_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-wide JobManager singleton
# ---------------------------------------------------------------------------
_manager_instance: Optional["JobManager"] = None
_manager_instance_lock = threading.Lock()


def get_manager() -> "JobManager":
    """Return the application-wide JobManager singleton.

    The instance is lazily created on first call and reused for the lifetime
    of the process.  This ensures all API requests share a single
    ``PriorityJobQueue`` and worker-thread pool instead of leaking new ones
    on every request.
    """
    global _manager_instance
    if _manager_instance is None:
        with _manager_instance_lock:
            if _manager_instance is None:
                from api.database import get_session_factory

                _manager_instance = JobManager(get_session_factory())
    return _manager_instance


# Priority mapping: lower number = higher priority (for heapq min-heap)
_PRIORITY_MAP = {"urgent": 0, "normal": 1, "low": 2}


class PriorityJobQueue:
    """Thread-safe priority queue for job processing.

    Jobs are ordered by priority (urgent > normal > low), then by
    submission timestamp (FIFO within the same priority level).
    """

    def __init__(self, max_concurrent: int = 4):
        self._queue: list[tuple[int, float, str]] = []  # (priority, timestamp, job_id)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._active = 0
        self._max_concurrent = max_concurrent
        self._shutdown = False

    def put(self, job_id: str, priority: str = "normal") -> None:
        """Add a job to the priority queue."""
        pri = _PRIORITY_MAP.get(priority, 1)
        with self._not_empty:
            heapq.heappush(self._queue, (pri, time.time(), job_id))
            self._not_empty.notify()

    def get(self) -> str | None:
        """Get the highest-priority job. Blocks until available or shutdown."""
        with self._not_empty:
            while not self._queue and not self._shutdown:
                self._not_empty.wait(timeout=1.0)
            if self._shutdown and not self._queue:
                return None
            if not self._queue:
                return None
            _, _, job_id = heapq.heappop(self._queue)
            self._active += 1
            return job_id

    def done(self) -> None:
        """Mark a job as completed (decrement active counter)."""
        with self._lock:
            self._active = max(0, self._active - 1)

    def shutdown(self) -> None:
        """Signal shutdown to all waiting workers."""
        with self._not_empty:
            self._shutdown = True
            self._not_empty.notify_all()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


def _coerce_processing_timeout_minutes(raw_value) -> Optional[int]:
    """Return a positive timeout override in minutes, or None if unset/invalid."""
    if raw_value is None:
        return None
    try:
        minutes = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid processing_timeout_minutes override: %r", raw_value)
        return None
    if minutes < 1:
        logger.warning("Ignoring non-positive processing_timeout_minutes override: %r", raw_value)
        return None
    return minutes


def _resolve_processing_timeout_minutes(settings: dict) -> int:
    """Resolve the effective processing timeout in minutes for a job."""
    override = _coerce_processing_timeout_minutes(
        settings.get("processing_timeout_minutes")
    )
    if override is not None:
        return override
    try:
        default_minutes = int(getattr(config, "JOB_PROCESSING_TIMEOUT_MINUTES", 30))
    except (TypeError, ValueError):
        default_minutes = 30
    return max(1, default_minutes)


def _format_processing_timeout(minutes: int) -> str:
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {unit}"


def _estimate_total_pages(path: Path) -> Optional[int]:
    """Best-effort page estimate for API progress reporting."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz

            with fitz.open(path) as doc:
                return max(1, int(doc.page_count))
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}:
            return 1
    except Exception:
        logger.debug("Failed to estimate page count for %s", path, exc_info=True)
    return None


def _count_text_outputs(output_dir: Path) -> int:
    text_dir = output_dir / "EXPORT" / "TEXT"
    if not text_dir.exists():
        return 0
    return len(list(text_dir.rglob("*.txt")))


def _has_completed_artifacts(output_dir: Path) -> bool:
    export_dir = output_dir / "EXPORT"
    if not export_dir.exists():
        return False
    pdf_dir = export_dir / "PDF"
    has_pdf = any(pdf_dir.rglob("*.pdf")) if pdf_dir.exists() else False
    has_text = _count_text_outputs(output_dir) > 0
    return has_pdf and has_text


def recover_interrupted_jobs(db_session_factory=None) -> int:
    """Reconcile jobs left active by an API restart or killed worker process."""
    if db_session_factory is None:
        from api.database import get_session_factory

        db_session_factory = get_session_factory()

    recovered = 0
    session: Session = db_session_factory()
    try:
        jobs = (
            session.query(Job)
            .filter(Job.status.in_(["submitted", "processing"]))
            .all()
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for job in jobs:
            output_dir = Path(job.result_path or Path(config.OUTPUT_FOLDER) / job.job_id)
            pages = _count_text_outputs(output_dir)
            if _has_completed_artifacts(output_dir):
                job.status = "completed"
                job.current_stage = "completed"
                job.error_message = None
                job.pages_completed = max(int(job.pages_completed or 0), pages)
                if not job.total_pages and (job.pages_completed or 0) > 0:
                    job.total_pages = job.pages_completed
            else:
                job.status = "failed"
                job.current_stage = "failed"
                job.error_message = "Job interrupted by API restart before completion"
            job.pid = None
            job.completed_at = now
            if job.started_at and job.processing_time is None:
                job.processing_time = max(0.0, (now - job.started_at).total_seconds())
            recovered += 1
        if recovered:
            session.commit()
            logger.info("Recovered %d interrupted OCR job(s) on startup", recovered)
        return recovered
    finally:
        session.close()


class JobManager:
    """Manages OCR job submission, tracking, and result retrieval."""

    def __init__(self, db_session_factory):
        self._session_factory = db_session_factory
        self._priority_queue = PriorityJobQueue(
            max_concurrent=int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))
        )
        self._workers_started = False
        self._workers_lock = threading.Lock()
        # track worker threads so shutdown() can join them gracefully.
        # Worker threads are non-daemon: relying on daemon=True killed in-flight
        # jobs without cleanup or audit trail at interpreter exit.
        self._worker_threads: list[threading.Thread] = []
        self._shutdown_called = False

    def _ensure_workers(self) -> None:
        """Start background worker threads if not already started."""
        if self._workers_started:
            return
        with self._workers_lock:
            if self._workers_started:
                return
            self._workers_started = True
            max_workers = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))
            for i in range(max_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"job-worker-{i}",
                    daemon=False,
                )
                t.start()
                self._worker_threads.append(t)

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Gracefully shut down worker threads.

        Signals the priority queue to stop, then waits up to ``timeout``
        seconds total for all worker threads to finish their in-flight jobs
        and exit.  Workers are non-daemon, so this method MUST be
        called at process shutdown to avoid orphaning in-flight jobs without
        cleanup or audit trail.

        Safe to call multiple times; subsequent calls are no-ops.

        Args:
            timeout: Maximum total seconds to wait for all workers to exit.

        Returns:
            True if all worker threads exited within the timeout; False
            otherwise.  A False return indicates at least one worker is
            still running an in-flight job past the deadline.
        """
        with self._workers_lock:
            if self._shutdown_called:
                return True
            self._shutdown_called = True
            threads = list(self._worker_threads)

        if not threads:
            # Workers never started; nothing to do.
            self._priority_queue.shutdown()
            return True

        logger.info(
            "JobManager.shutdown: signalling %d worker thread(s), timeout=%.1fs",
            len(threads),
            timeout,
        )
        # Signal the priority queue so any worker currently blocked in get()
        # wakes up and returns None, exiting the loop.
        self._priority_queue.shutdown()

        # Join each worker against a shared deadline so the aggregate wall
        # clock does not exceed ``timeout`` even if several workers need to
        # drain in-flight jobs.
        deadline = time.monotonic() + timeout
        all_exited = True
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                remaining = 0
            t.join(timeout=remaining)
            if t.is_alive():
                all_exited = False
                logger.warning(
                    "JobManager.shutdown: worker %s did not exit within timeout",
                    t.name,
                )

        if all_exited:
            logger.info("JobManager.shutdown: all worker threads exited cleanly")
        return all_exited

    def _worker_loop(self) -> None:
        """Worker thread that processes jobs from the priority queue."""
        while True:
            job_id = self._priority_queue.get()
            if job_id is None:
                break
            # Look up the job's source/output/settings before running pipeline
            session: Session = self._session_factory()
            try:
                job = session.get(Job, job_id)
                if not job:
                    logger.warning("Worker: job %s not found, skipping", job_id)
                    continue
                source_dir = str(Path(config.SOURCE_FOLDER) / job_id)
                output_dir = job.result_path or str(Path(config.OUTPUT_FOLDER) / job_id)
                settings = job.settings
            except Exception:
                logger.exception("Worker: failed to load job %s", job_id)
                continue
            finally:
                session.close()

            try:
                self._run_pipeline(job_id, source_dir, output_dir, settings)
            except Exception:
                logger.exception("Worker error processing job %s", job_id)
            finally:
                self._priority_queue.done()

    def _publish_job_event(self, job, event_type: str) -> None:
        """Persist and forward a job lifecycle event."""
        from api.events import publish_job_event

        publish_job_event(job, event_type)
        # signal waiting SSE generators to avoid polling delay
        try:
            from api.sse_notifier import notify_job_update

            notify_job_update(job.job_id)
        except Exception:
            pass  # Non-critical: SSE falls back to timeout polling

    def _fire_webhook(self, job) -> None:
        """Start webhook delivery for a job if a webhook URL is configured."""
        if not job.webhook_url:
            return
        from api.webhooks import start_webhook_delivery

        start_webhook_delivery(
            job.job_id,
            self._session_factory,
            webhook_timeout=config.WEBHOOK_TIMEOUT,
            webhook_max_retries=config.WEBHOOK_MAX_RETRIES,
            webhook_secret_default=config.WEBHOOK_SECRET,
        )

    def _check_batch_completion(self, job) -> None:
        """If the job belongs to a batch, check if the batch is complete."""
        batch_id = getattr(job, "batch_id", None)
        if not batch_id:
            return
        try:
            from api.batch_manager import BatchManager

            batch_mgr = BatchManager(self._session_factory, job_manager=self)
            batch_mgr.check_batch_completion(batch_id)
        except Exception:
            logger.exception(
                "Error checking batch completion for batch %s (job %s)",
                batch_id,
                job.job_id,
            )

    # ------------------------------------------------------------------
    # Dashboard collector wiring
    # ------------------------------------------------------------------

    def _update_dashboard_on_complete(self, job, session: Session) -> None:
        """Push job completion data to in-memory dashboard collectors.

        All calls are guarded so dashboard failures never break the
        pipeline.
        """
        _tenant = getattr(job, 'tenant_id', '') or ''
        try:
            from api.dashboard import get_collector

            collector = get_collector()
            collector.record_throughput(
                pages=job.pages_completed or 0,
                documents=1,
                bytes_processed=0,
                tenant_id=_tenant,
            )
            collector.record_latency(
                job_id=str(job.job_id),
                total_ms=int((job.processing_time or 0) * 1000),
                tenant_id=_tenant,
            )

            total = session.query(Job).count()
            active = session.query(Job).filter(
                Job.status.in_(["submitted", "processing"])
            ).count()
            completed = session.query(Job).filter(Job.status == "completed").count()
            failed = session.query(Job).filter(Job.status == "failed").count()
            queued = session.query(Job).filter(Job.status == "submitted").count()
            collector.update_job_counts(
                total=total,
                active=active,
                completed=completed,
                failed=failed,
                queued=queued,
                tenant_id=_tenant,
            )
        except Exception:
            logger.debug("Dashboard metrics collector update failed", exc_info=True)

        try:
            from api.analytics import get_analytics_store

            get_analytics_store().record_job(
                job_id=str(job.job_id),
                pages=job.pages_completed or 0,
                duration_seconds=job.processing_time or 0.0,
                success=(job.status == "completed"),
            )
        except Exception:
            logger.debug("Analytics store update failed", exc_info=True)

        try:
            from api.queue_alerting import get_queue_monitor

            active_count = session.query(Job).filter(
                Job.status.in_(["submitted", "processing"])
            ).count()
            get_queue_monitor().record_depth("processing", active_count)
        except Exception:
            logger.debug("Queue monitor update failed", exc_info=True)

        try:
            from sla_monitoring import get_monitor

            monitor = get_monitor()
            if monitor:
                _sla_tenant = _tenant or "default"
                _success = job.status == "completed"
                _latency = (job.processing_time or 0.0)  # already seconds
                monitor.record_request(_sla_tenant, _success, _latency)
        except Exception:
            logger.debug("SLA monitor feed failed", exc_info=True)

    def _update_dashboard_on_submit(self, active_count: int) -> None:
        """Push submission data to in-memory dashboard collectors."""
        try:
            from api.dashboard import get_collector

            get_collector().update_stage(
                "processing", active_workers=1, queue_depth=active_count
            )
        except Exception:
            logger.debug("Dashboard stage update on submit failed", exc_info=True)

        try:
            from api.queue_alerting import get_queue_monitor

            get_queue_monitor().record_depth("processing", active_count)
        except Exception:
            logger.debug("Queue monitor update on submit failed", exc_info=True)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        source_path: Optional[str] = None,
        upload_filename: Optional[str] = None,
        upload_content: Optional[bytes] = None,
        tenant_id: Optional[str] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ) -> Job:
        """Create a job and prepare its source folder."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_source = Path(config.SOURCE_FOLDER) / job_id
        job_output = Path(config.OUTPUT_FOLDER) / job_id
        job_source.mkdir(parents=True, exist_ok=True)
        job_output.mkdir(parents=True, exist_ok=True)

        # Resolve source file
        if upload_content and upload_filename:
            safe_name = os.path.basename(upload_filename)
            if not safe_name or safe_name.startswith('.'):
                raise ValueError(f"Invalid upload filename: {upload_filename!r}")
            dest = job_source / safe_name
            dest.write_bytes(upload_content)
            filename = safe_name
        elif source_path:
            src = validate_source_path_input(
                path_value=source_path,
                field_name="source_path",
                allowed_roots=(config.SOURCE_FOLDER,),
            )
            if not src.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")
            dest = job_source / src.name
            shutil.copy2(src, dest)
            filename = src.name
        else:
            raise ValueError("Either source_path or file upload is required.")

        settings = {
            "enable_docintel": enable_docintel,
            "docintel_mode": docintel_mode,
        }
        if processing_timeout_minutes is not None:
            settings["processing_timeout_minutes"] = processing_timeout_minutes
        source_size = dest.stat().st_size

        session: Session = self._session_factory()
        try:
            active = session.query(Job).filter(
                Job.status.in_(["submitted", "processing"])
            ).count()
            if active >= config.MAX_CONCURRENT_JOBS:
                session.close()
                raise ValueError("Job queue is at capacity")

            tenant = None
            if tenant_id:
                from api.quota import check_job_quota, check_storage_quota

                tenant = session.get(Tenant, tenant_id)
                if not tenant or tenant.status != "active":
                    raise ValueError(f"Tenant not found or inactive: {tenant_id}")
                check_job_quota(tenant, session=session)
                check_storage_quota(tenant, source_size, session=session)

            # Encrypt webhook secret before storing at rest (SEC-001)
            encrypted_secret = None
            if webhook_secret:
                from api.config import encrypt_webhook_secret

                encrypted_secret = encrypt_webhook_secret(webhook_secret)

            job = Job(
                job_id=job_id,
                status="submitted",
                priority=priority,
                source_file=filename,
                result_path=str(job_output),
                total_pages=_estimate_total_pages(dest),
                webhook_url=webhook_url,
                webhook_secret=encrypted_secret,
                tenant_id=tenant_id,
            )
            job.settings = settings
            session.add(job)
            session.flush()

            if tenant is not None:
                from api.usage import record_job_submitted, record_storage_used

                record_job_submitted(tenant_id, session=session)
                record_storage_used(tenant_id, source_size, session=session)

            session.commit()
            session.refresh(job)
            self._publish_job_event(job, "job.submitted")

            # Enqueue for priority-ordered background processing
            self._ensure_workers()
            self._priority_queue.put(job_id, priority=priority)

            self._update_dashboard_on_submit(active + 1)

            return job
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        job_id: str,
        source_dir: str,
        output_dir: str,
        settings: dict,
    ) -> None:
        """Run ocr_gpu_async.py as a subprocess for isolation."""
        session: Session = self._session_factory()
        try:
            job = session.get(Job, job_id)
            if not job:
                return

            job.status = "processing"
            job.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            session.refresh(job)
            self._publish_job_event(job, "job.processing")
            write_job_log(
                job_id,
                {
                    "level": "INFO",
                    "code": "JOB_STARTED",
                    "message": "Pipeline started",
                    "data": {
                        "source_dir": source_dir,
                        "output_dir": output_dir,
                        "enable_docintel": bool(settings.get("enable_docintel")),
                    },
                },
                base_dir=output_dir,
            )

            cmd = [
                "python",
                config.PIPELINE_SCRIPT,
                "--source", source_dir,
                "--output", output_dir,
            ]
            if settings.get("enable_docintel"):
                cmd.append("--enable-docintel")
                mode = settings.get("docintel_mode", "full")
                if mode != "full":
                    cmd.extend(["--docintel-mode", mode])

            logger.info("Starting pipeline for %s: %s", job_id, " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            job.pid = proc.pid
            session.commit()

            # Stream output to log, update progress via file system polling
            monitor = threading.Thread(
                target=self._monitor_progress,
                args=(job_id, output_dir, proc),
                daemon=True,
                name=f"monitor-{job_id}",
            )
            monitor.start()

            timeout_minutes = _resolve_processing_timeout_minutes(settings)
            timed_out = False
            try:
                proc.wait(timeout=timeout_minutes * 60)
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.error(
                    "Pipeline for %s exceeded timeout of %s",
                    job_id,
                    _format_processing_timeout(timeout_minutes),
                )
                try:
                    proc.kill()
                except Exception:
                    logger.exception("Failed to kill timed-out pipeline for %s", job_id)
                try:
                    proc.wait(timeout=10)
                except Exception:
                    logger.exception(
                        "Timed-out pipeline for %s did not exit cleanly after kill",
                        job_id,
                    )
            monitor.join(timeout=10)

            # Refresh and finalize
            session.refresh(job)
            if timed_out:
                job.status = "failed"
                job.error_message = (
                    "Job exceeded processing timeout "
                    f"({_format_processing_timeout(timeout_minutes)})"
                )
            elif proc.returncode == 0:
                try:
                    if job.tenant_id and (job.pages_completed or 0) > 0:
                        from api.quota import QuotaExceededError, check_page_quota
                        from api.usage import record_pages_processed

                        tenant = session.get(Tenant, job.tenant_id)
                        if tenant and tenant.status == "active":
                            check_page_quota(
                                tenant,
                                job.pages_completed or 0,
                                session=session,
                            )
                            record_pages_processed(
                                job.tenant_id,
                                job.pages_completed or 0,
                                session=session,
                            )
                except QuotaExceededError as exc:
                    job.status = "failed"
                    job.error_message = str(exc)
                else:
                    job.status = "completed"
                    job.current_stage = "completed"
                    if not job.total_pages and (job.pages_completed or 0) > 0:
                        job.total_pages = job.pages_completed
            else:
                job.status = "failed"
                job.current_stage = "failed"
                job.error_message = f"Pipeline exited with code {proc.returncode}"

            job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if job.started_at:
                delta = job.completed_at - job.started_at
                job.processing_time = delta.total_seconds()
                if job.tenant_id:
                    try:
                        from api.usage import record_processing_seconds

                        record_processing_seconds(
                            job.tenant_id,
                            job.processing_time,
                            period=job.started_at.strftime("%Y-%m"),
                            session=session,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to record tenant processing usage for %s",
                            job.tenant_id,
                        )
            job.pid = None
            session.commit()
            session.refresh(job)
            self._publish_job_event(job, f"job.{job.status}")

            if job.status == "completed":
                write_job_log(
                    job_id,
                    {
                        "level": "INFO",
                        "code": "JOB_COMPLETED",
                        "message": "Pipeline completed successfully",
                        "data": {
                            "pages_completed": job.pages_completed or 0,
                            "processing_time_seconds": job.processing_time,
                        },
                    },
                    base_dir=output_dir,
                )
            else:
                write_job_log(
                    job_id,
                    {
                        "level": "ERROR",
                        "code": "JOB_FAILED",
                        "message": job.error_message or "Pipeline failed",
                        "data": {"pages_completed": job.pages_completed or 0},
                    },
                    base_dir=output_dir,
                )

            self._update_dashboard_on_complete(job, session)

            self._fire_webhook(job)
            self._check_batch_completion(job)

        except Exception:
            logger.exception("Pipeline error for job %s", job_id)
            try:
                job = session.get(Job, job_id)
                if job:
                    job.status = "failed"
                    job.error_message = "Unexpected pipeline error"
                    job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    job.pid = None
                    session.commit()
                    session.refresh(job)
                    self._publish_job_event(job, "job.failed")
                    write_job_log(
                        job_id,
                        {
                            "level": "ERROR",
                            "code": "JOB_FAILED",
                            "message": "Unexpected pipeline error",
                        },
                        base_dir=output_dir,
                    )

                    self._update_dashboard_on_complete(job, session)

                    self._fire_webhook(job)
                    self._check_batch_completion(job)
            except Exception:
                logger.exception("Failed to update job status for %s", job_id)
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Progress monitoring
    # ------------------------------------------------------------------

    def _monitor_progress(
        self, job_id: str, output_dir: str, proc: subprocess.Popen
    ) -> None:
        """Poll output folder to estimate page progress."""
        text_dir = Path(output_dir) / "EXPORT" / "TEXT"
        session: Session = self._session_factory()
        try:
            while proc.poll() is None:
                time.sleep(config.PIPELINE_POLL_INTERVAL)
                if text_dir.exists():
                    pages = len(list(text_dir.rglob("*.txt")))
                    job = session.get(Job, job_id)
                    if job and pages != (job.pages_completed or 0):
                        job.pages_completed = pages
                        if not job.total_pages and pages > 0:
                            job.total_pages = pages
                        job.current_stage = "processing"
                        session.commit()
                        session.refresh(job)
                        self._publish_job_event(job, "job.progress")
                        write_job_log(
                            job_id,
                            {
                                "level": "INFO",
                                "code": "JOB_PROGRESS_TICK",
                                "message": f"Completed {pages} page(s)",
                                "data": {"pages_completed": pages},
                            },
                            base_dir=output_dir,
                        )
        except Exception:
            logger.exception("Progress monitor error for %s", job_id)
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def check_queue_capacity(self) -> bool:
        """Return True if the job queue can accept new jobs."""
        session = self._session_factory()
        try:
            active = session.query(Job).filter(
                Job.status.in_(["submitted", "processing"])
            ).count()
            return active < config.MAX_CONCURRENT_JOBS
        finally:
            session.close()

    def get_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[Job]:
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            return query.first()
        finally:
            session.close()

    def list_jobs(
        self,
        status: Optional[str] = None,
        batch_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        # Backward-compatible aliases (deprecated)
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        # Extended filters (D6)
        status_in: Optional[list[str]] = None,
        submitted_after: Optional[datetime] = None,
        submitted_before: Optional[datetime] = None,
        q: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> tuple[list[Job], int]:
        from sqlalchemy import case, or_

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
            query = session.query(Job)
            if status:
                query = query.filter(Job.status == status)
            if status_in:
                # Multi-status filter (D6).  Drop empty/invalid values.
                cleaned = [s for s in status_in if s]
                if cleaned:
                    query = query.filter(Job.status.in_(cleaned))
            if batch_id:
                query = query.filter(Job.batch_id == batch_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            if submitted_after is not None:
                query = query.filter(Job.created_at >= submitted_after)
            if submitted_before is not None:
                query = query.filter(Job.created_at <= submitted_before)
            if q:
                # Sanitize for LIKE injection: escape % and _ then wrap with %.
                # Match against job_id and source_file (case-insensitive).
                escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                pattern = f"%{escaped}%"
                query = query.filter(
                    or_(
                        Job.job_id.ilike(pattern, escape="\\"),
                        Job.source_file.ilike(pattern, escape="\\"),
                    )
                )
            total = query.count()

            # Sort selection
            if sort == "submitted_at_asc":
                ordering = (Job.created_at.asc(),)
            elif sort == "duration_desc":
                ordering = (Job.processing_time.desc().nullslast(), Job.created_at.desc())
            elif sort == "status":
                ordering = (Job.status.asc(), Job.created_at.desc())
            else:
                # Default: priority-then-time (urgent first, then newest first).
                # Also matches sort == "submitted_at_desc".
                priority_order = case(
                    (Job.priority == "urgent", 0),
                    (Job.priority == "normal", 1),
                    (Job.priority == "low", 2),
                    else_=1,
                )
                if sort == "submitted_at_desc":
                    ordering = (Job.created_at.desc(),)
                else:
                    ordering = (priority_order, Job.created_at.desc())

            jobs = (
                query.order_by(*ordering)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return jobs, total
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[Job]:
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            job = query.first()
            if not job:
                return None
            if job.status in ("completed", "failed", "cancelled"):
                return job

            # Kill subprocess if running
            if job.pid:
                try:
                    _sig = getattr(signal, "SIGKILL", 9)
                    os.kill(job.pid, _sig)
                except OSError:
                    pass

            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            job.pid = None
            session.commit()
            session.refresh(job)
            self._publish_job_event(job, "job.cancelled")

            self._fire_webhook(job)

            return job
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def retry_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[Job]:
        """Retry a failed or cancelled job by re-submitting with the same source."""
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            original = query.first()
            if not original:
                return None

            if original.status not in ("failed", "cancelled"):
                raise ValueError(
                    f"Only failed or cancelled jobs can be retried (current: {original.status})"
                )

            # Locate original source file
            original_source_dir = Path(config.SOURCE_FOLDER) / job_id
            if not original_source_dir.exists():
                raise FileNotFoundError(
                    f"Original source directory not found: {original_source_dir}"
                )

            source_files = list(original_source_dir.iterdir())
            if not source_files:
                raise FileNotFoundError("Original source file no longer available.")

            source_file = source_files[0]

            # Decrypt stored webhook secret before passing to submit()
            # (submit will re-encrypt for the new job record)
            retry_webhook_secret = None
            if original.webhook_secret:
                from api.config import decrypt_webhook_secret

                retry_webhook_secret = decrypt_webhook_secret(
                    original.webhook_secret
                )

            # Submit as a new job, copying the source file and preserving webhook
            return self.submit(
                source_path=str(source_file),
                tenant_id=original.tenant_id,
                priority=original.priority,
                enable_docintel=original.settings.get("enable_docintel", False),
                docintel_mode=original.settings.get("docintel_mode", "full"),
                processing_timeout_minutes=original.settings.get(
                    "processing_timeout_minutes"
                ),
                webhook_url=original.webhook_url,
                webhook_secret=retry_webhook_secret,
            )
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result_artifacts(self, job_id: str) -> dict[str, str]:
        """Return dict of artifact_type -> file_path for completed job.

        Each job processes a single input file, so each output subdirectory
        will contain at most one artifact of each type.
        """
        job = self.get_job(job_id)
        if not job or not job.result_path:
            return {}

        result_dir = Path(job.result_path)
        artifacts = {}

        pdf_dir = result_dir / "EXPORT" / "PDF"
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            if pdfs:
                artifacts["pdf"] = str(pdfs[0])

        text_dir = result_dir / "EXPORT" / "TEXT"
        if text_dir.exists():
            txts = list(text_dir.glob("*.txt"))
            if txts:
                artifacts["text"] = str(txts[0])

        struct_dir = result_dir / "EXPORT" / "STRUCTURE"
        if struct_dir.exists():
            jsons = list(struct_dir.glob("*.json"))
            if jsons:
                artifacts["structure"] = str(jsons[0])

        return artifacts
