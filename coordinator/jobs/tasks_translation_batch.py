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

import logging

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


# Default retry policy mirrors the existing translation worker pattern:
# fail-fast on policy denials, retry transient runtime errors a small
# number of times.
_DEFAULT_RETRIES = 2
_DEFAULT_RETRY_BACKOFF = 5.0  # seconds


@shared_task(
    bind=True,
    name='jobs.tasks_translation_batch.translate_batch_input',
    queue='translation_batch',
    autoretry_for=(RuntimeError,),
    retry_kwargs={'max_retries': _DEFAULT_RETRIES, 'countdown': _DEFAULT_RETRY_BACKOFF},
    acks_late=True,
)
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
    from .models import (
        BatchTranslationInput,
        BatchTranslationJob,
        CustodyEvent,
    )

    # Look up rows up front -- if either is missing we cannot proceed.
    try:
        job_row = BatchTranslationJob.objects.get(batch_id=batch_id)
        inp_row = BatchTranslationInput.objects.get(
            batch=job_row, client_ref=client_ref,
        )
    except (BatchTranslationJob.DoesNotExist,
            BatchTranslationInput.DoesNotExist) as exc:
        logger.error(
            "translate_batch_input: missing row for batch=%s ref=%s: %s",
            batch_id, client_ref, exc,
        )
        return {'status': 'missing', 'batch_id': batch_id, 'client_ref': client_ref}

    # If already cancelled or completed, no-op.
    if inp_row.status in ('cancelled', 'completed', 'failed'):
        return {'status': inp_row.status, 'batch_id': batch_id, 'client_ref': client_ref}

    inp_row.started_at = timezone.now()
    inp_row.status = 'running'
    inp_row.save(update_fields=['started_at', 'status'])

    try:
        result = _run_translation(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            tenant_id=tenant_id,
            glossary_enabled=glossary_enabled,
        )
    except Exception as exc:
        # Final-attempt failures fall through to the failure-write
        # branch below; transient retries are handled by autoretry_for.
        if self.request.retries < _DEFAULT_RETRIES:
            raise RuntimeError(str(exc)) from exc
        _record_failure(
            inp_row=inp_row,
            job_row=job_row,
            error_message=str(exc),
        )
        _emit_custody(
            CustodyEvent,
            event='BATCH_INPUT_FAILED',
            payload={
                'batch_id': batch_id,
                'client_ref': client_ref,
                'error': str(exc),
            },
        )
        _maybe_finalize(job_row, CustodyEvent)
        return {'status': 'failed', 'batch_id': batch_id, 'client_ref': client_ref}

    _record_success(inp_row=inp_row, job_row=job_row, result=result)
    _emit_custody(
        CustodyEvent,
        event='BATCH_INPUT_COMPLETED',
        payload={
            'batch_id': batch_id,
            'client_ref': client_ref,
            'engine_id': result['engine_id'],
            'char_count': result['char_count'],
        },
    )
    _maybe_finalize(job_row, CustodyEvent)
    return {
        'status': 'completed',
        'batch_id': batch_id,
        'client_ref': client_ref,
        'engine_id': result['engine_id'],
    }


def _run_translation(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    tenant_id: str,
    glossary_enabled: bool,
) -> dict:
    """Run glossary preprocessing + engine translation for one input.

    Returns a dict with keys ``target_text``, ``engine_id``,
    ``confidence``, ``glossary_hits``, ``char_count``.
    """
    from ocr_local.translation.router import (
        prepare_translation_input,
        select_engine_for_tenant,
    )

    # Glossary preprocessing must run BEFORE engine selection so
    # length-sensitive routing decisions see the modified text.
    if glossary_enabled:
        try:
            modified_text, hits = prepare_translation_input(
                tenant_id=tenant_id,
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        except Exception as exc:
            logger.debug(
                "translate_batch_input: glossary prep failed (non-fatal): %s",
                exc,
            )
            modified_text = text
            hits = []
    else:
        modified_text = text
        hits = []

    engine = select_engine_for_tenant(
        text=modified_text,
        source_lang=source_lang,
        target_lang=target_lang,
        tenant_id=tenant_id,
        allow_download=False,
    )

    spans = [{
        "span_id": "batch_0",
        "text": modified_text,
        "bbox": [0.0, 0.0, 100.0, 12.0],
    }]
    span_results = engine.translate_spans(spans, source_lang, target_lang)

    if span_results:
        target_text = span_results[0].target_text
        confidence = span_results[0].confidence
    else:
        target_text = modified_text
        confidence = None

    return {
        'target_text': target_text,
        'engine_id': engine.capability.id,
        'confidence': confidence,
        'glossary_hits': [
            {
                'term_id': h.term_id,
                'source_term': h.source_term,
                'target_term': h.target_term,
            }
            for h in hits
        ],
        'char_count': len(modified_text),
    }


def _record_success(*, inp_row, job_row, result: dict) -> None:
    inp_row.status = 'completed'
    inp_row.target_text = result['target_text']
    inp_row.engine_id = result['engine_id']
    inp_row.confidence = result['confidence']
    inp_row.glossary_hits_json = result['glossary_hits']
    inp_row.completed_at = timezone.now()
    inp_row.save(update_fields=[
        'status', 'target_text', 'engine_id', 'confidence',
        'glossary_hits_json', 'completed_at',
    ])
    type(job_row).objects.filter(pk=job_row.pk).update(
        completed_inputs=type(job_row).objects.filter(
            pk=job_row.pk
        ).values_list('completed_inputs', flat=True)[0] + 1,
    )


def _record_failure(*, inp_row, job_row, error_message: str) -> None:
    inp_row.status = 'failed'
    inp_row.error = error_message[:8000]
    inp_row.completed_at = timezone.now()
    inp_row.save(update_fields=['status', 'error', 'completed_at'])
    type(job_row).objects.filter(pk=job_row.pk).update(
        failed_inputs=type(job_row).objects.filter(
            pk=job_row.pk
        ).values_list('failed_inputs', flat=True)[0] + 1,
    )


def _emit_custody(CustodyEvent, *, event: str, payload: dict) -> None:
    """Persist a CustodyEvent row for the batch lifecycle.

    Uses the existing CustodyEvent model (chain-of-custody log) so the
    audit trail captures batch transitions alongside per-document
    events.  ``document_id`` is set to ``batch:<batch_id>`` so reviewers
    can reconstruct the batch lifecycle without colliding with per-doc
    events.
    """
    batch_id = str(payload.get('batch_id', ''))
    document_id = f"batch:{batch_id}" if batch_id else 'batch:unknown'
    try:
        CustodyEvent.objects.create(
            document_id=document_id,
            job=None,
            event_type=event,
            data=payload,
        )
    except Exception as exc:
        logger.debug(
            "translate_batch_input: CustodyEvent insert failed (non-fatal): %s",
            exc,
        )


def _maybe_finalize(job_row, CustodyEvent) -> None:
    """If all inputs are terminal, flip job to ``completed``."""
    from .models import BatchTranslationInput, BatchTranslationJob

    job_row.refresh_from_db(fields=['status', 'total_inputs'])
    if job_row.status in BatchTranslationJob.TERMINAL_STATUSES:
        return
    pending_remaining = BatchTranslationInput.objects.filter(
        batch=job_row, status__in=['pending', 'running'],
    ).count()
    if pending_remaining > 0:
        return
    completed = BatchTranslationInput.objects.filter(
        batch=job_row, status='completed',
    ).count()
    failed = BatchTranslationInput.objects.filter(
        batch=job_row, status='failed',
    ).count()
    job_row.status = (
        BatchTranslationJob.STATUS_COMPLETED
        if failed == 0
        else BatchTranslationJob.STATUS_COMPLETED  # partial success still completed
    )
    job_row.completed_at = timezone.now()
    job_row.save(update_fields=['status', 'completed_at'])
    _emit_custody(
        CustodyEvent,
        event='BATCH_COMPLETED',
        payload={
            'batch_id': job_row.batch_id,
            'completed_count': completed,
            'failed_count': failed,
        },
    )
