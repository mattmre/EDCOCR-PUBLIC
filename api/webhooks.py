"""Webhook delivery engine for job completion notifications.

Delivers HMAC-signed JSON payloads to user-specified callback URLs
when jobs reach terminal states (completed, failed, cancelled).
Uses stdlib urllib.request to avoid adding external dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from ocr_distributed.ssrf import (
    safe_opener as _safe_opener,
)
from ocr_distributed.ssrf import (
    validate_webhook_url,
)
from ocr_local.config.version import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL validation — delegated to ocr_distributed.ssrf (shared SSRF module)
# ---------------------------------------------------------------------------

# Thread pool for webhook delivery (bounded concurrency)
_webhook_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")
_webhook_pool_lock = threading.Lock()


def _get_webhook_pool() -> ThreadPoolExecutor:
    """Return a live webhook thread pool, recreating it if needed."""
    global _webhook_pool
    with _webhook_pool_lock:
        if getattr(_webhook_pool, "_shutdown", False):
            _webhook_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")
        return _webhook_pool


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


def compute_signature(payload_json: str, secret: str, timestamp: int) -> str:
    """Compute HMAC-SHA256 signature for a webhook payload.

    Signs the string: ``{timestamp}.{payload_json}``
    Returns: ``sha256={hex_digest}``
    """
    message = f"{timestamp}.{payload_json}"
    digest = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _load_pii_entities(job_id: str) -> list[dict]:
    """Load PII entities from the coordinator database if available.

    Attempts to import the Django PiiEntity model and query for entities
    associated with the given job ID. Returns an empty list if the
    coordinator models are not available (e.g., standalone API mode).

    Parameters
    ----------
    job_id : str
        The job ID to load entities for.

    Returns
    -------
    list of dict
        Each dict contains entity_type, confidence_score, page_index,
        and bounding_box fields.
    """
    try:
        from jobs.models import PiiEntity  # Django coordinator model

        entities = PiiEntity.objects.filter(job_id=job_id).select_related("page")
        return [
            {
                "entity_type": e.entity_type,
                "confidence_score": round(e.confidence, 4),
                "page_index": e.page.page_num if e.page else 0,
                "bounding_box": e.bounding_box if isinstance(e.bounding_box, list) else [],
            }
            for e in entities
        ]
    except Exception:
        # Coordinator models not available (standalone API mode) or query error
        return []


def build_webhook_payload(
    job,
    event: str,
    *,
    enrich_entities: bool = False,
) -> dict:
    """Build webhook payload from a job database object.

    Parameters
    ----------
    job : api.database.Job
        The job record (must be attached to a session or have attributes loaded).
    event : str
        One of ``job.completed``, ``job.failed``, ``job.cancelled``.
    enrich_entities : bool
        When True and the job has completed successfully, include
        PII/PHI entity bounding boxes in the payload.

    Returns
    -------
    dict
        Structured payload ready for JSON serialization.
    """
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "status": job.status,
        "source_file": job.source_file,
        "processing": {
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time_seconds": job.processing_time,
            "pages_completed": job.pages_completed or 0,
            "total_pages": job.total_pages or 0,
        },
        "error_message": job.error_message if job.status == "failed" else None,
    }

    # Enrich with PII entity bounding boxes when enabled and job is completed
    if enrich_entities and event == "job.completed":
        entities = _load_pii_entities(job.job_id)
        if entities:
            payload["entities"] = entities

    return payload


# ---------------------------------------------------------------------------
# Delivery engine
# ---------------------------------------------------------------------------

# Retry delays in seconds (exponential backoff: 5, 10, 20, 40)
_RETRY_DELAYS = [5, 10, 20, 40]


def deliver_webhook(
    job_id: str,
    session_factory,
    *,
    webhook_timeout: int = 30,
    webhook_max_retries: int = 3,
    webhook_secret_default: str = "",
) -> None:
    """Deliver a webhook notification with retry logic.

    Runs synchronously (intended for use in a background thread).
    Reads the job from the database, builds the payload, signs it,
    POSTs to the webhook_url, and updates webhook_status in the DB.
    """
    session = session_factory()
    try:
        from api.database import Job

        job = session.get(Job, job_id)
        if not job or not job.webhook_url:
            return

        # SSRF Protection: Re-validate URL at delivery time (TOCTOU mitigation)
        from api.config import WEBHOOK_ALLOW_HTTP, WEBHOOK_ALLOW_PRIVATE

        try:
            validate_webhook_url(
                job.webhook_url,
                allow_http=WEBHOOK_ALLOW_HTTP,
                allow_private=WEBHOOK_ALLOW_PRIVATE,
            )
        except ValueError as exc:
            logger.warning(
                "Webhook for job %s failed re-validation before delivery: %s",
                job_id, exc,
            )
            job.webhook_status = "failed"
            job.webhook_last_error = f"URL validation failed: {exc}"
            session.commit()
            return

        # Determine event type
        status_to_event = {
            "completed": "job.completed",
            "failed": "job.failed",
            "cancelled": "job.cancelled",
        }
        event = status_to_event.get(job.status)
        if not event:
            logger.warning(
                "Webhook skipped for job %s: unexpected status %s",
                job_id,
                job.status,
            )
            return

        # Build and serialize payload (enrich with entities if configured)
        enrich = False
        try:
            from api.config import WEBHOOK_ENRICH_ENTITIES

            enrich = WEBHOOK_ENRICH_ENTITIES
        except (ImportError, AttributeError):
            pass
        payload = build_webhook_payload(job, event, enrich_entities=enrich)
        payload_json = json.dumps(payload, separators=(",", ":"))

        # Compute signature (decrypt stored secret first — SEC-001)
        stored_secret = job.webhook_secret
        if stored_secret:
            from api.config import decrypt_webhook_secret

            stored_secret = decrypt_webhook_secret(stored_secret)
        secret = stored_secret or webhook_secret_default

        # Static headers — defined once (signature/timestamp refreshed per attempt)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"OCR-Pipeline-Webhook/{__version__}",
            "X-Webhook-Event": event,
            "X-Webhook-Job-ID": job_id,
        }

        # Mark as pending
        job.webhook_status = "pending"
        session.commit()

        # Attempt delivery with retries
        max_attempts = 1 + min(webhook_max_retries, len(_RETRY_DELAYS))
        last_error: Optional[str] = None

        for attempt in range(max_attempts):
            # Refresh HMAC timestamp and signature on every attempt so that
            # receivers enforcing timestamp freshness accept retries.
            timestamp = int(time.time())
            if secret:
                headers["X-Webhook-Signature"] = compute_signature(
                    payload_json, secret, timestamp,
                )
                headers["X-Webhook-Timestamp"] = str(timestamp)

            try:
                req = urllib.request.Request(
                    job.webhook_url,
                    data=payload_json.encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with _safe_opener.open(req, timeout=webhook_timeout) as resp:
                    status_code = resp.getcode()

                if 200 <= status_code < 300:
                    # Success
                    job.webhook_status = "delivered"
                    job.webhook_attempts = attempt + 1
                    job.webhook_last_error = None
                    session.commit()
                    logger.info(
                        "Webhook delivered for job %s (attempt %d, status %d)",
                        job_id,
                        attempt + 1,
                        status_code,
                    )
                    return
                else:
                    last_error = f"HTTP {status_code}"
                    logger.warning(
                        "Webhook delivery for job %s returned %d (attempt %d/%d)",
                        job_id,
                        status_code,
                        attempt + 1,
                        max_attempts,
                    )
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                last_error = str(exc)[:500]
                logger.warning(
                    "Webhook delivery error for job %s (attempt %d/%d): %s",
                    job_id,
                    attempt + 1,
                    max_attempts,
                    last_error,
                )

            # Update attempt count in DB
            job.webhook_attempts = attempt + 1
            job.webhook_last_error = last_error
            session.commit()

            # Wait before retry (unless this was the last attempt)
            if attempt < max_attempts - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                time.sleep(delay)

        # All retries exhausted
        job.webhook_status = "failed"
        job.webhook_attempts = max_attempts
        job.webhook_last_error = last_error
        session.commit()
        logger.error(
            "Webhook delivery failed for job %s after %d attempts: %s",
            job_id,
            max_attempts,
            last_error,
        )

        # Add to dead-letter queue for later inspection/retry
        try:
            from api.webhook_dlq import add_to_dlq

            add_to_dlq(
                job_id=job_id,
                webhook_url=job.webhook_url,
                event_type=event,
                payload=payload,
                last_error=last_error or "Unknown error",
                attempts=max_attempts,
            )
        except Exception as dlq_exc:
            # Log DLQ write failures explicitly (never silently drop)
            logger.error(
                "Webhook DLQ write failed for job %s: %s", job_id, dlq_exc,
                exc_info=True,
            )

    except Exception:
        logger.exception("Unexpected error in webhook delivery for job %s", job_id)
        try:
            from api.database import Job as _Job

            job = session.get(_Job, job_id)
            if job:
                job.webhook_status = "failed"
                job.webhook_last_error = "Internal delivery error"
                session.commit()
        except Exception:
            logger.exception("Failed to update webhook status for job %s", job_id)
    finally:
        session.close()


def start_webhook_delivery(
    job_id: str,
    session_factory,
    *,
    webhook_timeout: int = 30,
    webhook_max_retries: int = 3,
    webhook_secret_default: str = "",
) -> None:
    """Launch webhook delivery in a daemon thread.

    Parameters
    ----------
    job_id : str
        The job ID to deliver a webhook for.
    session_factory : sessionmaker
        SQLAlchemy session factory for database access.
    webhook_timeout : int
        HTTP request timeout in seconds.
    webhook_max_retries : int
        Maximum number of retry attempts.
    webhook_secret_default : str
        Fallback HMAC secret when no per-job secret is set.
    """
    try:
        _get_webhook_pool().submit(
            deliver_webhook,
            job_id,
            session_factory,
            webhook_timeout=webhook_timeout,
            webhook_max_retries=webhook_max_retries,
            webhook_secret_default=webhook_secret_default,
        )
    except RuntimeError:
        # Executor can be unavailable during interpreter shutdown;
        # deliver inline instead of dropping the notification.
        logger.warning(
            "Webhook thread pool unavailable; delivering job %s synchronously.",
            job_id,
        )
        deliver_webhook(
            job_id,
            session_factory,
            webhook_timeout=webhook_timeout,
            webhook_max_retries=webhook_max_retries,
            webhook_secret_default=webhook_secret_default,
        )


# ---------------------------------------------------------------------------
# Batch webhook support
# ---------------------------------------------------------------------------


def build_batch_webhook_payload(
    batch,
    jobs: list,
    event: str,
) -> dict:
    """Build webhook payload for a batch completion event.

    Parameters
    ----------
    batch : api.database.Batch
        The batch record.
    jobs : list of api.database.Job
        Child jobs belonging to the batch.
    event : str
        One of ``batch.completed``, ``batch.partial_failure``,
        ``batch.failed``, ``batch.cancelled``.

    Returns
    -------
    dict
        Structured payload ready for JSON serialization.
    """
    return {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch.batch_id,
        "status": batch.status,
        "total_jobs": batch.total_jobs,
        "jobs_completed": batch.jobs_completed,
        "jobs_failed": batch.jobs_failed,
        "jobs_cancelled": batch.jobs_cancelled,
        "processing_time_seconds": batch.processing_time,
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "source_file": j.source_file,
            }
            for j in jobs
        ],
    }


def deliver_batch_webhook(
    batch_id: str,
    session_factory,
    *,
    webhook_timeout: int = 30,
    webhook_max_retries: int = 3,
    webhook_secret_default: str = "",
) -> None:
    """Deliver a webhook notification for a batch.

    Runs synchronously (intended for use in a background thread).
    """
    session = session_factory()
    try:
        from api.database import Batch, Job

        batch = session.get(Batch, batch_id)
        if not batch or not batch.webhook_url:
            return

        # SSRF Protection
        from api.config import WEBHOOK_ALLOW_HTTP, WEBHOOK_ALLOW_PRIVATE

        try:
            validate_webhook_url(
                batch.webhook_url,
                allow_http=WEBHOOK_ALLOW_HTTP,
                allow_private=WEBHOOK_ALLOW_PRIVATE,
            )
        except ValueError as exc:
            logger.warning(
                "Batch webhook for %s failed URL validation: %s",
                batch_id, exc,
            )
            batch.webhook_status = "failed"
            batch.webhook_last_error = f"URL validation failed: {exc}"
            session.commit()
            return

        # Determine event type
        status_to_event = {
            "completed": "batch.completed",
            "partial_failure": "batch.partial_failure",
            "failed": "batch.failed",
            "cancelled": "batch.cancelled",
        }
        event = status_to_event.get(batch.status)
        if not event:
            logger.warning(
                "Batch webhook skipped for %s: unexpected status %s",
                batch_id, batch.status,
            )
            return

        # Build payload
        jobs = (
            session.query(Job)
            .filter(Job.batch_id == batch_id)
            .all()
        )
        payload = build_batch_webhook_payload(batch, jobs, event)
        payload_json = json.dumps(payload, separators=(",", ":"))

        # Compute signature (decrypt stored secret first — SEC-001)
        stored_secret = batch.webhook_secret
        if stored_secret:
            from api.config import decrypt_webhook_secret

            stored_secret = decrypt_webhook_secret(stored_secret)
        secret = stored_secret or webhook_secret_default

        # Static headers — defined once (signature/timestamp refreshed per attempt)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"OCR-Pipeline-Webhook/{__version__}",
            "X-Webhook-Event": event,
            "X-Webhook-Batch-ID": batch_id,
        }

        batch.webhook_status = "pending"
        session.commit()

        # Attempt delivery with retries
        max_attempts = 1 + min(webhook_max_retries, len(_RETRY_DELAYS))
        last_error: Optional[str] = None

        for attempt in range(max_attempts):
            # Refresh HMAC timestamp and signature on every attempt so that
            # receivers enforcing timestamp freshness accept retries.
            timestamp = int(time.time())
            if secret:
                headers["X-Webhook-Signature"] = compute_signature(
                    payload_json, secret, timestamp,
                )
                headers["X-Webhook-Timestamp"] = str(timestamp)

            try:
                req = urllib.request.Request(
                    batch.webhook_url,
                    data=payload_json.encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with _safe_opener.open(req, timeout=webhook_timeout) as resp:
                    status_code = resp.getcode()

                if 200 <= status_code < 300:
                    batch.webhook_status = "delivered"
                    batch.webhook_attempts = attempt + 1
                    batch.webhook_last_error = None
                    session.commit()
                    logger.info(
                        "Batch webhook delivered for %s (attempt %d, status %d)",
                        batch_id, attempt + 1, status_code,
                    )
                    return
                else:
                    last_error = f"HTTP {status_code}"
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                last_error = str(exc)[:500]
                logger.warning(
                    "Batch webhook error for %s (attempt %d/%d): %s",
                    batch_id, attempt + 1, max_attempts, last_error,
                )

            batch.webhook_attempts = attempt + 1
            batch.webhook_last_error = last_error
            session.commit()

            if attempt < max_attempts - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                time.sleep(delay)

        batch.webhook_status = "failed"
        batch.webhook_attempts = max_attempts
        batch.webhook_last_error = last_error
        session.commit()
        logger.error(
            "Batch webhook failed for %s after %d attempts: %s",
            batch_id, max_attempts, last_error,
        )

        # Add to dead-letter queue for later inspection/retry
        try:
            from api.webhook_dlq import add_to_dlq

            add_to_dlq(
                job_id=batch_id,
                webhook_url=batch.webhook_url,
                event_type=event,
                payload=payload,
                last_error=last_error or "Unknown error",
                attempts=max_attempts,
            )
        except Exception as dlq_exc:
            # Log DLQ write failures explicitly (never silently drop)
            logger.error(
                "Webhook DLQ write failed for job %s: %s", batch_id, dlq_exc,
                exc_info=True,
            )

    except Exception:
        logger.exception(
            "Unexpected error in batch webhook delivery for %s", batch_id
        )
        try:
            from api.database import Batch as _Batch

            batch = session.get(_Batch, batch_id)
            if batch:
                batch.webhook_status = "failed"
                batch.webhook_last_error = "Internal delivery error"
                session.commit()
        except Exception:
            logger.exception(
                "Failed to update batch webhook status for %s", batch_id
            )
    finally:
        session.close()


def start_batch_webhook_delivery(
    batch_id: str,
    session_factory,
    *,
    webhook_timeout: int = 30,
    webhook_max_retries: int = 3,
    webhook_secret_default: str = "",
) -> None:
    """Launch batch webhook delivery in the webhook thread pool."""
    try:
        _get_webhook_pool().submit(
            deliver_batch_webhook,
            batch_id,
            session_factory,
            webhook_timeout=webhook_timeout,
            webhook_max_retries=webhook_max_retries,
            webhook_secret_default=webhook_secret_default,
        )
    except RuntimeError:
        logger.warning(
            "Webhook thread pool unavailable; delivering batch %s synchronously.",
            batch_id,
        )
        deliver_batch_webhook(
            batch_id,
            session_factory,
            webhook_timeout=webhook_timeout,
            webhook_max_retries=webhook_max_retries,
            webhook_secret_default=webhook_secret_default,
        )
