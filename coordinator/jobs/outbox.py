"""Durable database-to-Celery task publication.

The outbox deliberately implements at-least-once publication.  The broker
cannot participate in the coordinator database transaction, so a relay crash
after publication may produce a duplicate message.  Consumers must therefore
use their existing job/page ownership CAS, which makes a duplicate harmless
and observable instead of allowing a second durable execution.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from typing import Callable

from celery import current_app
from django.db import IntegrityError, connection, transaction
from django.db.models import CharField, F, Q
from django.db.models.functions import Cast
from django.utils import timezone

from .models import Job, TaskDispatchOutbox
from .rls import control_plane_rls_context

logger = logging.getLogger(__name__)

MAX_OUTBOX_PAYLOAD_BYTES = 64 * 1024
DEFAULT_OUTBOX_LEASE_SECONDS = 120
MAX_OUTBOX_LEASE_SECONDS = 3_600
MAX_OUTBOX_RETRY_SECONDS = 3_600
DEFAULT_OUTBOX_MAX_ATTEMPTS = 12
MAX_OUTBOX_MAX_ATTEMPTS = 100
_TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled")


def _bounded_payload(value, *, field: str, default):
    """Validate JSON task payloads before they enter a durable row."""
    if value is None:
        value = default
    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"outbox {field} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_OUTBOX_PAYLOAD_BYTES:
        raise ValueError(f"outbox {field} exceeds {MAX_OUTBOX_PAYLOAD_BYTES} bytes")
    return value


def enqueue_task_dispatch(
    *,
    job: Job,
    stage: str,
    generation: str,
    task_name: str,
    args=None,
    kwargs=None,
    queue: str = "",
    priority: int = 0,
    max_attempts: int = DEFAULT_OUTBOX_MAX_ATTEMPTS,
) -> TaskDispatchOutbox:
    """Insert or return the idempotent dispatch contract for a job stage.

    Callers must invoke this inside the transaction that changes the job
    state.  Reusing the same key is safe and returns the original contract.
    """
    if not isinstance(stage, str) or not stage or len(stage) > 64:
        raise ValueError("outbox stage must be a non-empty string of at most 64 chars")
    if not isinstance(generation, str) or not generation or len(generation) > 64:
        raise ValueError(
            "outbox generation must be a non-empty string of at most 64 chars"
        )
    if not isinstance(task_name, str) or not task_name or len(task_name) > 255:
        raise ValueError("outbox task_name is invalid")
    try:
        priority = int(priority)
    except (TypeError, ValueError) as exc:
        raise ValueError("outbox priority must be an integer") from exc
    if not 0 <= priority <= 9:
        raise ValueError("outbox priority must be between 0 and 9")
    try:
        max_attempts = int(max_attempts)
    except (TypeError, ValueError) as exc:
        raise ValueError("outbox max_attempts must be an integer") from exc
    if not 1 <= max_attempts <= MAX_OUTBOX_MAX_ATTEMPTS:
        raise ValueError(
            f"outbox max_attempts must be between 1 and {MAX_OUTBOX_MAX_ATTEMPTS}"
        )
    args = _bounded_payload(args, field="args", default=[])
    kwargs = _bounded_payload(kwargs, field="kwargs", default={})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ValueError("outbox args must be a list and kwargs must be an object")

    try:
        row, _created = TaskDispatchOutbox.objects.get_or_create(
            job=job,
            stage=stage,
            generation=generation,
            defaults={
                "task_name": task_name,
                "args": args,
                "kwargs": kwargs,
                "queue": queue or "",
                "priority": priority,
                "max_attempts": max_attempts,
            },
        )
    except IntegrityError:
        # The unique constraint is the authority when two coordinators race
        # on a backend that reports an IntegrityError after the first insert.
        row = TaskDispatchOutbox.objects.get(
            job=job,
            stage=stage,
            generation=generation,
        )

    if (
        row.task_name != task_name
        or row.args != args
        or row.kwargs != kwargs
        or row.queue != (queue or "")
        or row.priority != priority
        or row.max_attempts != max_attempts
    ):
        raise ValueError(
            "existing outbox key has a different task contract; refusing mutation"
        )
    return row


def _lease_seconds(value: int | None) -> int:
    if value is None:
        value = DEFAULT_OUTBOX_LEASE_SECONDS
    return max(1, min(MAX_OUTBOX_LEASE_SECONDS, int(value)))


def _claim_dispatch(
    *,
    dispatch_id=None,
    lease_seconds: int | None = None,
    exclude_job_ids=(),
) -> TaskDispatchOutbox | None:
    """Atomically claim one pending or expired dispatch row."""
    now = timezone.now()
    token = uuid.uuid4().hex
    lease_until = now + timedelta(seconds=_lease_seconds(lease_seconds))
    eligible = (
        Q(status=TaskDispatchOutbox.Status.PENDING)
        | Q(
            status=TaskDispatchOutbox.Status.FAILED,
            available_at__lte=now,
        )
        | Q(
            status=TaskDispatchOutbox.Status.CLAIMED,
            lease_expires_at__lt=now,
        )
    )

    with transaction.atomic():
        query = (
            TaskDispatchOutbox.objects.select_for_update()
            .filter(eligible)
            .exclude(job__status__in=_TERMINAL_JOB_STATUSES)
            .filter(
                generation=Cast(
                    F("job__dispatch_generation"),
                    output_field=CharField(),
                )
            )
        )
        if exclude_job_ids:
            query = query.exclude(job_id__in=exclude_job_ids)
        if dispatch_id is not None:
            query = query.filter(pk=dispatch_id)
        if connection.vendor == "postgresql":
            query = query.select_for_update(skip_locked=True)
        row = query.order_by("-priority", "created_at", "dispatch_id").first()
        if row is None:
            return None
        row.status = TaskDispatchOutbox.Status.CLAIMED
        row.claim_token = token
        row.lease_expires_at = lease_until
        row.updated_at = now
        row.save(
            update_fields=["status", "claim_token", "lease_expires_at", "updated_at"]
        )
    return row


def _retry_delay(attempts: int) -> int:
    return min(MAX_OUTBOX_RETRY_SECONDS, 5 * (2 ** min(max(attempts - 1, 0), 12)))


def resolve_task_dispatch_queue(row: TaskDispatchOutbox, kwargs=None) -> str:
    """Resolve topology-dependent OCR routing immediately before publish."""
    from coordinator.celery import resolve_ocr_dispatch_queue

    return resolve_ocr_dispatch_queue(
        row.task_name,
        row.queue,
        kwargs=kwargs or {},
    )


def _default_dispatcher(row: TaskDispatchOutbox):
    """Publish through Celery's registered task without importing task code."""
    job = Job.objects.get(pk=row.job_id)
    if str(row.generation) != str(job.dispatch_generation):
        raise RuntimeError("refusing stale-generation outbox publication")
    if row.task_name == "jobs.tasks_layoutlm.run_layoutlm_extraction":
        # This optional module is not found by the conventional tasks.py
        # autodiscovery name. Import it only for its explicit allowlisted
        # contract, after Django setup, so publishers can resolve the task.
        from . import tasks_layoutlm  # noqa: F401

    task = current_app.tasks.get(row.task_name)
    if task is None:
        raise RuntimeError(f"Celery task is not registered: {row.task_name}")
    kwargs = dict(row.kwargs or {})
    kwargs["dispatch_generation"] = int(row.generation)
    if row.task_name in {
        "jobs.tasks.process_document",
        "jobs.tasks.process_text_only",
        "jobs.tasks.process_page",
        "jobs.tasks.render_page",
        "jobs.tasks.infer_page",
        "jobs.tasks.finalize_page",
    }:
        from .presigned import (
            create_presigned_capability,
            create_stage_capability,
            is_presigned_mode,
        )

        if is_presigned_mode():
            operation = {
                "jobs.tasks.process_document": "process_document",
                "jobs.tasks.process_text_only": "process_text_only",
                "jobs.tasks.process_page": "process_page",
                "jobs.tasks.render_page": "render_page",
                "jobs.tasks.infer_page": "infer_page",
                "jobs.tasks.finalize_page": "finalize_page",
            }[row.task_name]
            if operation in {"render_page", "infer_page", "finalize_page"}:
                from .models import PageOCRStage

                stage = PageOCRStage.objects.get(
                    job=job,
                    page_num=int(row.args[1]),
                    generation=int(row.generation),
                )
                kwargs["presigned_capability_id"] = create_stage_capability(
                    job, stage, operation
                )
            else:
                kwargs["presigned_capability_id"] = create_presigned_capability(
                    job, operation, {}
                )
    elif row.task_name == "jobs.tasks_layoutlm.run_layoutlm_extraction":
        from .presigned import (
            create_presigned_capability,
            is_presigned_mode,
        )

        if is_presigned_mode():
            kwargs["presigned_capability_id"] = create_presigned_capability(
                job, "layoutlm", {}
            )
    options = {
        "queue": resolve_task_dispatch_queue(row, kwargs),
        "priority": row.priority,
    }
    if kwargs:
        return task.apply_async(args=list(row.args or []), kwargs=kwargs, **options)
    return task.apply_async(args=list(row.args or []), **options)


def _mark_publish_failed(
    row: TaskDispatchOutbox, token: str, exc: Exception
) -> str | None:
    now = timezone.now()
    message = str(exc).strip()[:8_000] or exc.__class__.__name__
    with transaction.atomic():
        current = TaskDispatchOutbox.objects.filter(
            pk=row.pk,
            status=TaskDispatchOutbox.Status.CLAIMED,
            claim_token=token,
        ).first()
        if current is None:
            return None
        attempts = current.attempts + 1
        exhausted = attempts >= current.max_attempts
        current.status = (
            TaskDispatchOutbox.Status.DEAD
            if exhausted
            else TaskDispatchOutbox.Status.FAILED
        )
        current.attempts = attempts
        current.available_at = (
            now if exhausted else now + timedelta(seconds=_retry_delay(attempts))
        )
        current.dead_at = now if exhausted else None
        current.claim_token = ""
        current.lease_expires_at = None
        current.last_error = message
        current.save(
            update_fields=[
                "status",
                "attempts",
                "available_at",
                "claim_token",
                "lease_expires_at",
                "last_error",
                "dead_at",
                "updated_at",
            ]
        )
        return current.status


def _mark_published(
    row: TaskDispatchOutbox,
    token: str,
    broker_task_id: str,
) -> bool:
    now = timezone.now()
    with transaction.atomic():
        updated = TaskDispatchOutbox.objects.filter(
            pk=row.pk,
            status=TaskDispatchOutbox.Status.CLAIMED,
            claim_token=token,
        ).update(
            status=TaskDispatchOutbox.Status.PUBLISHED,
            attempts=F("attempts") + 1,
            broker_task_id=broker_task_id,
            published_at=now,
            claim_token="",
            lease_expires_at=None,
            last_error="",
            updated_at=now,
        )
        if updated != 1:
            return False
        if broker_task_id:
            # A worker can consume the message before this update.  The
            # conditional write preserves a worker-owned task id if it won
            # the race, while still recording the broker id for a blank row.
            Job.objects.filter(
                pk=row.job_id,
                celery_task_id="",
                dispatch_generation=row.generation,
            ).update(celery_task_id=broker_task_id)
    return True


def _publish_task_dispatch_outbox_control_plane(
    *,
    limit: int = 100,
    dispatch_id=None,
    dispatcher: Callable[[TaskDispatchOutbox], object] | None = None,
) -> list[dict]:
    """Relay a bounded number of contracts with at-least-once semantics."""
    limit = max(1, min(1_000, int(limit)))
    results: list[dict] = []
    seen_job_ids: set[object] = set()
    for _ in range(limit):
        row = _claim_dispatch(
            dispatch_id=dispatch_id,
            exclude_job_ids=() if dispatch_id is not None else seen_job_ids,
        )
        if row is None and seen_job_ids and dispatch_id is None:
            seen_job_ids.clear()
            row = _claim_dispatch()
        if row is None:
            break
        seen_job_ids.add(row.job_id)
        token = row.claim_token
        try:
            result = (dispatcher or _default_dispatcher)(row)
            broker_task_id = str(getattr(result, "id", "") or "")
        except Exception as exc:  # broker failure is durable, not swallowed
            failure_status = _mark_publish_failed(row, token, exc)
            if failure_status == TaskDispatchOutbox.Status.DEAD:
                from .tasks import _terminalize_dead_dispatch

                _terminalize_dead_dispatch(row.pk)
            logger.warning(
                "Task dispatch outbox publish failed id=%s task=%s: %s",
                row.dispatch_id,
                row.task_name,
                exc,
            )
            results.append(
                {
                    "dispatch_id": str(row.dispatch_id),
                    "status": failure_status or TaskDispatchOutbox.Status.CLAIMED,
                    "error": str(exc),
                }
            )
        else:
            published = _mark_published(row, token, broker_task_id)
            status = (
                TaskDispatchOutbox.Status.PUBLISHED
                if published
                else TaskDispatchOutbox.Status.CLAIMED
            )
            results.append(
                {
                    "dispatch_id": str(row.dispatch_id),
                    "status": status,
                    "broker_task_id": broker_task_id,
                }
            )
        if dispatch_id is not None:
            break
    return results


def publish_task_dispatch_outbox(
    *,
    limit: int = 100,
    dispatch_id=None,
    dispatcher: Callable[[TaskDispatchOutbox], object] | None = None,
) -> list[dict]:
    """Publish durable rows under the explicit cross-tenant relay policy."""
    with control_plane_rls_context():
        return _publish_task_dispatch_outbox_control_plane(
            limit=limit,
            dispatch_id=dispatch_id,
            dispatcher=dispatcher,
        )


__all__ = [
    "DEFAULT_OUTBOX_LEASE_SECONDS",
    "DEFAULT_OUTBOX_MAX_ATTEMPTS",
    "MAX_OUTBOX_PAYLOAD_BYTES",
    "enqueue_task_dispatch",
    "publish_task_dispatch_outbox",
    "resolve_task_dispatch_queue",
]
