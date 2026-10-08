"""Celery tasks for batch translation scheduling (Plan B Wave M2 -- B17).

Tasks run on the dedicated ``translation_batch`` queue so the batch
workload can be scaled independently of the OCR/NLP queues.

The task body is intentionally small:

1. Resolve glossary preprocessing via
   :func:`ocr_local.translation.router.prepare_translation_input`.
2. Select an engine via
   :func:`ocr_local.translation.router.select_engine_for_tenant`.
3. Run the engine and write the result back to the
   ``BatchTranslationInput`` row.
4. On success, emit ``BATCH_INPUT_COMPLETED``; on terminal failure,
   emit ``BATCH_INPUT_FAILED`` and store the error string.
5. When the batch's pending/running counts drop to zero, flip the
   parent ``BatchTranslationJob`` status to ``completed`` and emit
   ``BATCH_COMPLETED``.

All heavy imports (``ocr_local.translation``) are inside the task body
so the module can be imported (and registered with Celery via
``autodiscover_tasks``) without pulling translation engines into the
coordinator process.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import timedelta
from functools import wraps

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


# Default retry policy mirrors the existing translation worker pattern:
# fail-fast on policy denials, retry transient runtime errors a small
# number of times.
_DEFAULT_RETRIES = 2
_DEFAULT_RETRY_BACKOFF = 5.0  # seconds
_DEFAULT_INPUT_LEASE_SECONDS = 30 * 60
_MAX_INPUT_LEASE_SECONDS = 24 * 60 * 60


def _tenant_bound_translation_task(func):
    """Pin the worker connection before the batch/input ORM lookup."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        from .rls import worker_tenant_rls_context

        with worker_tenant_rls_context(kwargs.get("tenant_id")):
            return func(self, *args, **kwargs)

    return wrapper


def _input_lease_seconds() -> int:
    """Return a bounded lease used to reclaim crashed translation inputs."""
    raw = os.environ.get(
        "TRANSLATION_INPUT_LEASE_SECONDS",
        str(_DEFAULT_INPUT_LEASE_SECONDS),
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_INPUT_LEASE_SECONDS
    if not math.isfinite(value) or value < 1 or value > _MAX_INPUT_LEASE_SECONDS:
        return _DEFAULT_INPUT_LEASE_SECONDS
    return int(value)


@shared_task(
    bind=True,
    name="jobs.tasks_translation_batch.translate_batch_input",
    queue="translation_batch",
    autoretry_for=(RuntimeError,),
    retry_kwargs={"max_retries": _DEFAULT_RETRIES, "countdown": _DEFAULT_RETRY_BACKOFF},
    acks_late=True,
)
@_tenant_bound_translation_task
def translate_batch_input(
    self,
    *,
    batch_id: str,
    client_ref: str,
    text: str,
    source_lang: str,
    target_lang: str,
    tenant_id: str,
    glossary_enabled: bool = True,
):
    """Translate a single input within a batch.

    On success: writes ``target_text``, ``engine_id``, ``confidence``,
    ``glossary_hits_json`` to the corresponding
    :class:`BatchTranslationInput` row and increments
    ``completed_inputs`` on the parent
    :class:`BatchTranslationJob`.

    On terminal failure (after autoretry exhaustion): writes the error
    string to the row, increments ``failed_inputs``, and emits
    ``BATCH_INPUT_FAILED``.
    """
    # Lazy imports -- these may fail in environments without translation
    # adapters installed (CPU-only worker images), in which case the
    # task records the error and exits cleanly.
    from django.db import transaction

    from .models import (
        BatchTranslationInput,
        BatchTranslationJob,
        CustodyEvent,
    )

    task_token = str(
        getattr(self.request, "id", None) or f"translation:{batch_id}:{client_ref}"
    )

    # Claim the input under a row lock before doing any external translation.
    # A Celery redelivery or duplicate publish must not run the same client_ref
    # concurrently and then race the terminal accounting update.
    try:
        with transaction.atomic():
            job_row = BatchTranslationJob.objects.select_for_update().get(
                batch_id=batch_id
            )
            inp_row = BatchTranslationInput.objects.select_for_update().get(
                batch=job_row, client_ref=client_ref
            )
            if inp_row.status in ("cancelled", "completed", "failed"):
                # A crash after committing the terminal input but before an
                # older worker finalized the parent is repaired by the
                # broker's acks-late redelivery while both rows are locked.
                _maybe_finalize(job_row, CustodyEvent)
                return {
                    "status": inp_row.status,
                    "batch_id": batch_id,
                    "client_ref": client_ref,
                }
            if job_row.status in job_row.TERMINAL_STATUSES:
                return {
                    "status": job_row.status,
                    "batch_id": batch_id,
                    "client_ref": client_ref,
                }
            if inp_row.status == "running":
                current_owner = str(inp_row.celery_task_id or "")
                started_at = inp_row.started_at
                lease_expired = (
                    started_at is not None
                    and timezone.now() - started_at
                    >= timedelta(seconds=_input_lease_seconds())
                )
                if current_owner != task_token and not lease_expired:
                    return {
                        "status": "running",
                        "batch_id": batch_id,
                        "client_ref": client_ref,
                    }

            # The broker payload is not an authority for tenant identity. Bind
            # policy/cache lookup to the tenant stored on the batch row.
            persisted_tenant_id = str(job_row.tenant_id or "").strip()
            if not persisted_tenant_id:
                raise ValueError("translation batch has no persisted tenant binding")
            if tenant_id and str(tenant_id).strip() != persisted_tenant_id:
                raise PermissionError(
                    "translation task tenant binding does not match batch"
                )
            tenant_id = persisted_tenant_id
            # Broker arguments are routing hints, not translation authority.
            # A forged/redelivered payload must not substitute text, language,
            # or glossary policy for the durable batch/input contract.
            persisted_contract = {
                "text": inp_row.text,
                "source_lang": job_row.source_lang,
                "target_lang": job_row.target_lang,
                "glossary_enabled": bool(job_row.glossary_enabled),
            }
            supplied_contract = {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "glossary_enabled": bool(glossary_enabled),
            }
            if supplied_contract != persisted_contract:
                logger.warning(
                    "translation broker contract differed from persisted row; "
                    "batch=%s ref=%s",
                    batch_id,
                    client_ref,
                )
            text = persisted_contract["text"]
            source_lang = persisted_contract["source_lang"]
            target_lang = persisted_contract["target_lang"]
            glossary_enabled = persisted_contract["glossary_enabled"]
            privilege_flag = bool(
                (inp_row.optional_metadata_json or {}).get("privilege_flag", False)
            )
            inp_row.started_at = timezone.now()
            inp_row.status = "running"
            inp_row.celery_task_id = task_token
            inp_row.save(update_fields=["started_at", "status", "celery_task_id"])
            if job_row.status == BatchTranslationJob.STATUS_PENDING:
                job_row.status = BatchTranslationJob.STATUS_RUNNING
                job_row.save(update_fields=["status"])
    except (
        BatchTranslationJob.DoesNotExist,
        BatchTranslationInput.DoesNotExist,
    ) as exc:
        logger.error(
            "translate_batch_input: missing row for batch=%s ref=%s: %s",
            batch_id,
            client_ref,
            exc,
        )
        return {"status": "missing", "batch_id": batch_id, "client_ref": client_ref}

    try:
        result = _run_translation(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            tenant_id=tenant_id,
            glossary_enabled=glossary_enabled,
            privilege_flag=privilege_flag,
        )
    except Exception as exc:
        # Final-attempt failures fall through to the failure-write
        # branch below; transient retries are handled by autoretry_for.
        if self.request.retries < _DEFAULT_RETRIES:
            raise RuntimeError(str(exc)) from exc
        recorded = _record_failure(
            inp_row=inp_row,
            job_row=job_row,
            error_message=str(exc),
            custody_model=CustodyEvent,
            custody_payload={
                "batch_id": batch_id,
                "tenant_id": job_row.tenant_id,
                "client_ref": client_ref,
                "error": str(exc),
            },
            task_token=task_token,
        )
        if not recorded:
            return {
                "status": "already_terminal",
                "batch_id": batch_id,
                "client_ref": client_ref,
            }
        return {"status": "failed", "batch_id": batch_id, "client_ref": client_ref}

    if not _record_success(
        inp_row=inp_row,
        job_row=job_row,
        result=result,
        custody_model=CustodyEvent,
        custody_payload={
            "batch_id": batch_id,
            "tenant_id": job_row.tenant_id,
            "client_ref": client_ref,
            "engine_id": result["engine_id"],
            "char_count": result["char_count"],
            "source_sha256": result["source_sha256"],
            "glossary_hash": result["glossary_hash"],
            "weights_sha256": result["weights_sha256"],
            "engine_license": result["engine_license"],
            "runtime_version": result["runtime_version"],
        },
        task_token=task_token,
    ):
        return {
            "status": "already_terminal",
            "batch_id": batch_id,
            "client_ref": client_ref,
        }
    return {
        "status": "completed",
        "batch_id": batch_id,
        "client_ref": client_ref,
        "engine_id": result["engine_id"],
    }


def _run_translation(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    tenant_id: str,
    glossary_enabled: bool,
    privilege_flag: bool = False,
) -> dict:
    """Run glossary preprocessing + engine translation for one input.

    Returns a dict with keys ``target_text``, ``engine_id``,
    ``confidence``, ``glossary_hits``, ``char_count``.
    """
    import hashlib

    from ocr_local.translation.glossary import (
        apply_glossary,
        compute_glossary_hash,
        load_tenant_glossary,
    )
    from ocr_local.translation.router import select_engine_for_tenant

    # Glossary preprocessing must run BEFORE engine selection so
    # length-sensitive routing decisions see the modified text.
    if glossary_enabled:
        entries = load_tenant_glossary(tenant_id, source_lang, target_lang)
        modified_text, hits = apply_glossary(text, entries, span_index=0)
        glossary_hash = compute_glossary_hash(entries)
    else:
        modified_text = text
        hits = []
        glossary_hash = hashlib.sha256(b"[]").hexdigest()

    engine = select_engine_for_tenant(
        text=modified_text,
        source_lang=source_lang,
        target_lang=target_lang,
        tenant_id=tenant_id,
        allow_download=False,
        privilege_flag=privilege_flag,
    )

    spans = [
        {
            "span_id": "batch_0",
            "text": modified_text,
            "bbox": [0.0, 0.0, 100.0, 12.0],
        }
    ]
    span_results = engine.translate_spans(spans, source_lang, target_lang)
    provenance = engine.validated_provenance()

    if span_results:
        target_text = span_results[0].target_text
        confidence = span_results[0].confidence
    else:
        target_text = modified_text
        confidence = None

    return {
        "target_text": target_text,
        "engine_id": engine.capability.id,
        "confidence": confidence,
        "glossary_hits": [
            {
                "term_id": h.term_id,
                "source_term": h.source_term,
                "target_term": h.target_term,
            }
            for h in hits
        ],
        "char_count": len(modified_text),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "glossary_hash": glossary_hash,
        "weights_sha256": provenance.weights_sha256,
        "engine_license": provenance.license,
        "runtime_version": provenance.runtime_version,
    }


def _record_success(
    *,
    inp_row,
    job_row,
    result: dict,
    custody_model,
    custody_payload: dict,
    task_token: str,
) -> bool:
    """Commit one success exactly once and return whether it won the claim."""
    from django.db import transaction
    from django.db.models import F

    with transaction.atomic():
        locked_input = type(inp_row).objects.select_for_update().get(pk=inp_row.pk)
        if (
            locked_input.status != "running"
            or str(locked_input.celery_task_id or "") != task_token
        ):
            return False
        locked_input.status = "completed"
        locked_input.target_text = result["target_text"]
        locked_input.engine_id = result["engine_id"]
        locked_input.confidence = result["confidence"]
        locked_input.glossary_hits_json = result["glossary_hits"]
        locked_input.completed_at = timezone.now()
        locked_input.save(
            update_fields=[
                "status",
                "target_text",
                "engine_id",
                "confidence",
                "glossary_hits_json",
                "completed_at",
            ]
        )
        # Atomic increment: several Celery workers can finish inputs from the
        # same batch concurrently; read-then-write loses completions under load.
        type(job_row).objects.filter(pk=job_row.pk).update(
            completed_inputs=F("completed_inputs") + 1,
        )
        _emit_custody(
            custody_model,
            event="BATCH_INPUT_COMPLETED",
            payload=custody_payload,
        )
        # Keep terminal input accounting and the last-input parent transition
        # in the same database commit. A worker crash can no longer persist an
        # all-terminal input set beneath a RUNNING parent.
        _maybe_finalize(job_row, custody_model)
    return True


def _record_failure(
    *,
    inp_row,
    job_row,
    error_message: str,
    custody_model,
    custody_payload: dict,
    task_token: str,
) -> bool:
    """Commit one terminal failure exactly once and return claim status."""
    from django.db import transaction
    from django.db.models import F

    with transaction.atomic():
        locked_input = type(inp_row).objects.select_for_update().get(pk=inp_row.pk)
        if (
            locked_input.status != "running"
            or str(locked_input.celery_task_id or "") != task_token
        ):
            return False
        locked_input.status = "failed"
        locked_input.error = error_message[:8000]
        locked_input.completed_at = timezone.now()
        locked_input.save(update_fields=["status", "error", "completed_at"])
        # See _record_success: failure accounting must also be atomic.
        type(job_row).objects.filter(pk=job_row.pk).update(
            failed_inputs=F("failed_inputs") + 1,
        )
        _emit_custody(
            custody_model,
            event="BATCH_INPUT_FAILED",
            payload=custody_payload,
        )
        _maybe_finalize(job_row, custody_model)
    return True


def _emit_custody(_custody_model, *, event: str, payload: dict) -> None:
    """Persist a CustodyEvent row for the batch lifecycle.

    Uses the existing CustodyEvent model (chain-of-custody log) so the
    audit trail captures batch transitions alongside per-document
    events.  ``document_id`` is set to ``batch:<batch_id>`` so reviewers
    can reconstruct the batch lifecycle without colliding with per-doc
    events.
    """
    batch_id = str(payload.get("batch_id", ""))
    document_id = f"batch:{batch_id}" if batch_id else "batch:unknown"
    if len(document_id) > 64:
        document_id = (
            "batch:" + hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:58]
        )
    from .tasks import _record_control_plane_custody_event

    _record_control_plane_custody_event(
        tenant_id=str(payload.get("tenant_id") or "").strip(),
        document_id=document_id,
        event_type=event,
        data=payload,
        worker_hostname="translation-batch-worker",
    )


def _maybe_finalize(job_row, CustodyEvent) -> None:
    """If all inputs are terminal, flip job to ``completed``."""
    from django.db import transaction

    from .models import BatchTranslationInput, BatchTranslationJob

    with transaction.atomic():
        locked_job = BatchTranslationJob.objects.select_for_update().get(pk=job_row.pk)
        if locked_job.status in BatchTranslationJob.TERMINAL_STATUSES:
            return
        pending_remaining = BatchTranslationInput.objects.filter(
            batch=locked_job,
            status__in=["pending", "running"],
        ).count()
        if pending_remaining > 0:
            return
        completed = BatchTranslationInput.objects.filter(
            batch=locked_job,
            status="completed",
        ).count()
        failed = BatchTranslationInput.objects.filter(
            batch=locked_job,
            status="failed",
        ).count()
        updated = (
            type(locked_job)
            .objects.filter(
                pk=locked_job.pk,
                status__in=["pending", "running"],
            )
            .update(
                status=BatchTranslationJob.STATUS_COMPLETED,
                completed_at=timezone.now(),
            )
        )
        if not updated:
            return
        _emit_custody(
            CustodyEvent,
            event="BATCH_COMPLETED",
            payload={
                "batch_id": locked_job.batch_id,
                "tenant_id": locked_job.tenant_id,
                "completed_count": completed,
                "failed_count": failed,
            },
        )
