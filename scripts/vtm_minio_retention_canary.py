#!/usr/bin/env python3
"""Exercise tenant purge failure ordering against a fresh local MinIO job.

The only destructive target is a UUID-named tenant created by this run in the
isolated OCR_LOCAL kind lab. Four rollback/partial-delete boundaries are
injected, the real purge command is retried, and the S3-compatible object
prefix plus PostgreSQL rows are checked at the end. This is local evidence;
it does not establish production S3/IAM or cross-system atomicity.

Required environment: ``OCR_COORDINATOR_API_KEY`` (local lab key; never logged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import vtm_broker_outage_canary as broker  # noqa: E402
import vtm_pg8_worker_recovery_canary as pg8  # noqa: E402
from vtm_canary_safety import (  # noqa: E402
    require_fresh_directory,
    require_isolated_kind_identity,
    require_loopback_http_url,
    write_json_no_replace,
)

FAULT_ASSERTIONS = (
    "dry_run_reports_one_job_and_two_pages",
    "generation_fence_failure_rolls_back",
    "outbox_fence_failure_rolls_back",
    "custody_insert_failure_rolls_back_fence_and_outbox",
    "partial_minio_delete_leaves_fenced_retryable_job",
    "pending_outbox_is_dead_before_storage_delete",
    "start_custody_precedes_minio_delete",
    "retry_reestablishes_monotonic_fence_and_removes_remaining_objects",
    "final_tenant_rows_and_minio_prefix_removed",
    "purge_audit_events_retained",
)

FAULT_BOUNDARIES = (
    "generation_fence",
    "outbox_fence",
    "start_custody_commit",
    "partial_minio_object_delete",
)

SAFE_FAILURE_REASONS = frozenset(
    {
        "coordinator_health_failed",
        "Ready API and coordinator runtime identities are required",
        "coordinator_submit_failed",
        "coordinator_submit_result_invalid",
        "fixture_page_count_unavailable",
        "Django state query failed",
        "Django state query returned no JSON",
        "job_did_not_complete",
        "completed retention job lacks a Ready source-bound worker identity",
        "live_minio_job_or_output_preflight_failed",
    }
)


def safe_failure_reason(exc: BaseException) -> str:
    """Keep a bounded, non-sensitive reason while dropping free-form diagnostics."""
    if isinstance(exc, urllib.error.URLError):
        return "coordinator_transport_unavailable"
    if isinstance(exc, TimeoutError):
        return "operation_timed_out"
    message = str(exc)
    if type(exc) is RuntimeError and message in SAFE_FAILURE_REASONS:
        return message
    return "unclassified_failure"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--kubeconfig", required=True, type=Path)
    parser.add_argument("--namespace", default="ocr-local")
    parser.add_argument("--release", default="ocr-kind")
    parser.add_argument("--api-url", default="http://127.0.0.1:18089")
    parser.add_argument(
        "--fixture", type=Path, default=Path("tests/fixtures/test-canary-2page.pdf")
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser.parse_args(argv)


def _exec_coordinator_python(
    kubeconfig: Path, namespace: str, deployment: str, source: str
) -> dict | None:
    result = broker.command(
        [
            *broker.kube(kubeconfig),
            "exec",
            "-n",
            namespace,
            f"deployment/{deployment}",
            "-c",
            "coordinator",
            "--",
            "python3",
            "-c",
            source,
        ],
        timeout=240,
    )
    if not result.get("ok"):
        return None
    lines = [
        line
        for line in result.get("stdout", "").splitlines()
        if line.strip().startswith("{")
    ]
    if not lines:
        return None
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def safe_fault_matrix(value: dict | None) -> dict:
    """Retain only fixed assertions and boundary names from remote output.

    The remote purge command can surface broker errors, database diagnostics,
    filenames, and exception text. None of those free-form values belong in a
    retained certification artifact.
    """
    raw_assertions = value.get("assertions") if isinstance(value, dict) else None
    if not isinstance(raw_assertions, dict):
        raw_assertions = {}
    assertions = {name: raw_assertions.get(name) is True for name in FAULT_ASSERTIONS}

    raw_faults = value.get("faults") if isinstance(value, dict) else None
    observed = []
    if isinstance(raw_faults, list):
        for row in raw_faults:
            if isinstance(row, dict) and row.get("boundary") in FAULT_BOUNDARIES:
                observed.append(row["boundary"])
    boundaries = [name for name in FAULT_BOUNDARIES if observed.count(name) == 1]
    ok = (
        isinstance(value, dict)
        and value.get("ok") is True
        and all(assertions.values())
        and len(boundaries) == len(FAULT_BOUNDARIES)
    )
    result = {"ok": ok, "assertions": assertions, "fault_boundaries": boundaries}
    if isinstance(value, dict):
        before = _safe_snapshot(value.get("before"))
        final_snapshot = _safe_snapshot(value.get("final_snapshot"))
        retry_boundary = value.get("retry_boundary")
        retry_boundary = retry_boundary if isinstance(retry_boundary, dict) else {}
        result["before"] = before
        result["final_snapshot"] = final_snapshot
        result["retry_boundary"] = {
            "job_state": _safe_job(retry_boundary.get("job_state")),
            "objects_before_delete": _bounded_count(
                retry_boundary.get("objects_before_delete")
            ),
        }
        safe_faults = []
        for row in (
            value.get("faults", []) if isinstance(value.get("faults"), list) else []
        ):
            if not isinstance(row, dict) or row.get("boundary") not in FAULT_BOUNDARIES:
                continue
            fault = {
                "boundary": row["boundary"],
                "attempt": {
                    "raised": (row.get("attempt") or {}).get("raised") is True
                    if isinstance(row.get("attempt"), dict)
                    else False
                },
                "state_unchanged": row.get("state_unchanged") is True,
            }
            if row["boundary"] == "partial_minio_object_delete":
                fault["snapshot"] = _safe_snapshot(row.get("snapshot"))
            safe_faults.append(fault)
        result["faults"] = safe_faults
        fault_by_boundary = {row["boundary"]: row for row in safe_faults}
        rollback_ok = all(
            fault_by_boundary.get(name, {}).get("attempt", {}).get("raised") is True
            and fault_by_boundary.get(name, {}).get("state_unchanged") is True
            for name in FAULT_BOUNDARIES[:3]
        )
        partial = fault_by_boundary.get("partial_minio_object_delete", {}).get(
            "snapshot", {}
        )
        partial_dead = any(
            row.get("stage") == "retention-canary-pending"
            and row.get("status") == "dead"
            for row in partial.get("dispatches", [])
        )
        partial_ok = (
            partial.get("job", {}).get("status") == "cancelled"
            and partial_dead
            and len(partial.get("object_keys", []))
            == len(before.get("object_keys", [])) - 1
        )
        final_events = set(final_snapshot.get("purge_events", []))
        shape_ok = (
            before.get("tenant_jobs") == 1
            and (before.get("pages") or 0) > 0
            and bool(before.get("object_keys"))
            and rollback_ok
            and partial_ok
            and result["retry_boundary"].get("job_state", {}).get("status")
            == "cancelled"
            and (result["retry_boundary"].get("objects_before_delete") or 0) > 0
            and final_snapshot.get("tenant_jobs") == 0
            and final_snapshot.get("pages") == 0
            and final_snapshot.get("tenant_dispatch_rows") == 0
            and final_snapshot.get("tenant_job_custody_rows") == 0
            and final_snapshot.get("object_keys") == []
            and {
                "tenant_purge_started",
                "tenant_purge_job_storage_deleted",
                "tenant_purged",
            }.issubset(final_events)
        )
        result["ok"] = result["ok"] and shape_ok
    if not result["ok"]:
        result["failure"] = (
            "assertion_failed"
            if isinstance(value, dict)
            else "remote_response_unavailable"
        )
    return result


def _bounded_count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10_000_000 else None


def _safe_job(value: object) -> dict:
    row = value if isinstance(value, dict) else {}
    status = row.get("status")
    return {
        "status": status
        if isinstance(status, str)
        and status in {"queued", "processing", "completed", "failed", "cancelled"}
        else "unknown",
        "dispatch_generation": _bounded_count(row.get("dispatch_generation")),
        "pages_completed": _bounded_count(row.get("pages_completed")),
        "total_pages": _bounded_count(row.get("total_pages")),
    }


def _safe_snapshot(value: object) -> dict:
    row = value if isinstance(value, dict) else {}
    key_values = row.get("object_keys")
    key_hashes = []
    if isinstance(key_values, list) and len(key_values) <= 10_000:
        key_hashes = [
            hashlib.sha256(item.encode("utf-8", errors="replace")).hexdigest()
            for item in key_values
            if isinstance(item, str)
        ]
    dispatches = []
    raw_dispatches = row.get("dispatches")
    if isinstance(raw_dispatches, list):
        for dispatch in raw_dispatches[:10_000]:
            if not isinstance(dispatch, dict):
                continue
            dispatches.append(
                {
                    "stage": (
                        "retention-canary-pending"
                        if dispatch.get("stage") == "retention-canary-pending"
                        else "other"
                    ),
                    "status": (
                        dispatch.get("status")
                        if isinstance(dispatch.get("status"), str)
                        and dispatch.get("status")
                        in {"pending", "queued", "sent", "failed", "dead"}
                        else "unknown"
                    ),
                }
            )
    allowed_events = {
        "tenant_purge_started",
        "tenant_purge_job_storage_deleted",
        "tenant_purged",
    }
    raw_events = row.get("purge_events")
    return {
        "tenant_jobs": _bounded_count(row.get("tenant_jobs")),
        "pages": _bounded_count(row.get("pages")),
        "pii": _bounded_count(row.get("pii")),
        "tenant_dispatch_rows": _bounded_count(row.get("tenant_dispatch_rows")),
        "tenant_job_custody_rows": _bounded_count(row.get("tenant_job_custody_rows")),
        "purge_events": [
            event
            for event in raw_events
            if isinstance(event, str) and event in allowed_events
        ]
        if isinstance(raw_events, list)
        else [],
        "job": _safe_job(row.get("job")),
        "dispatches": dispatches,
        "object_keys": key_hashes,
    }


def safe_active_artifacts(value: dict | None) -> dict:
    """Keep only bounded artifact checks and require the intended S3 backend."""
    value = value if isinstance(value, dict) else {}
    pdf = value.get("pdf") if isinstance(value.get("pdf"), dict) else {}
    text = value.get("text") if isinstance(value.get("text"), dict) else {}

    def positive_int(row: dict, key: str) -> int | None:
        item = row.get(key)
        return item if type(item) is int and 0 < item <= 100_000_000 else None

    revision = value.get("revision")
    if not (
        isinstance(revision, (str, int))
        and not isinstance(revision, bool)
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", str(revision))
    ):
        revision = None

    def safe_digest(row: dict, key: str) -> str | None:
        digest = row.get(key)
        return (
            digest
            if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
            else None
        )

    safe_pdf = {
        "matches_active_pointer": pdf.get("matches_active_pointer") is True,
        "size_bytes": positive_int(pdf, "size_bytes"),
        "page_count": positive_int(pdf, "page_count"),
        "pdf_text_characters": positive_int(pdf, "pdf_text_characters"),
        "sha256": safe_digest(pdf, "sha256"),
        "sha256_expected": safe_digest(pdf, "sha256_expected"),
    }
    safe_text = {
        "matches_active_pointer": text.get("matches_active_pointer") is True,
        "size_bytes": positive_int(text, "size_bytes"),
        "text_characters": positive_int(text, "text_characters"),
        "sha256": safe_digest(text, "sha256"),
        "sha256_expected": safe_digest(text, "sha256_expected"),
    }
    result = {
        "backend": value.get("backend")
        if isinstance(value.get("backend"), str)
        and value.get("backend") in {"s3", "nfs"}
        else "unknown",
        "revision": revision,
        "pdf": safe_pdf,
        "text": safe_text,
    }
    result["ok"] = bool(
        value.get("ok") is True
        and result["backend"] == "s3"
        and result["revision"]
        and safe_pdf["matches_active_pointer"]
        and safe_text["matches_active_pointer"]
        and safe_pdf["size_bytes"]
        and safe_text["size_bytes"]
        and safe_pdf["page_count"]
        and safe_pdf["pdf_text_characters"]
        and safe_pdf["sha256"]
        and safe_pdf["sha256"] == safe_pdf["sha256_expected"]
        and safe_text["text_characters"]
        and safe_text["sha256"]
        and safe_text["sha256"] == safe_text["sha256_expected"]
    )
    return result


def safe_job_before_purge(value: object) -> dict:
    row = value if isinstance(value, dict) else {}
    job_id = row.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        job_id,
    ):
        job_id = None
    status = row.get("status")
    source_hash = row.get("source_hash")
    return {
        "job_id": job_id,
        "status": status
        if isinstance(status, str)
        and status in {"queued", "processing", "completed", "failed", "cancelled"}
        else "unknown",
        "total_pages": _bounded_count(row.get("total_pages")),
        "pages_completed": _bounded_count(row.get("pages_completed")),
        "pages_failed": _bounded_count(row.get("pages_failed")),
        "source_hash": source_hash
        if isinstance(source_hash, str) and re.fullmatch(r"[a-f0-9]{64}", source_hash)
        else None,
    }


def observe_job_and_capture_worker(
    kubeconfig: Path,
    namespace: str,
    coordinator: str,
    job_id: str,
    runtime_images: dict,
    evidence: dict,
    timeout_sec: int,
) -> dict | None:
    """Observe job state while sampling the short-lived KEDA worker Pod.

    Capture before the database query because that query can outlast a fast OCR
    job; retry after a non-terminal observation in case KEDA became Ready while
    the query was running.
    """
    deadline = time.monotonic() + timeout_sec
    final_state = None
    terminal_states = {"completed", "failed", "cancelled"}
    while time.monotonic() < deadline:
        if not runtime_images.get("worker"):
            broker.capture_worker_runtime_identity(
                kubeconfig, namespace, evidence["release"], runtime_images
            )
        final_state = pg8.django_state(kubeconfig, namespace, coordinator, job_id)
        if not isinstance(final_state, dict) or not isinstance(
            final_state.get("job"), dict
        ):
            raise RuntimeError("Django state query returned no JSON")
        evidence["observed_job_state"] = safe_job_before_purge(final_state["job"])
        if final_state["job"].get("status") in terminal_states:
            page_rows = final_state.get("pages")
            page_rows = page_rows if isinstance(page_rows, list) else []
            evidence["page_rows_before_purge"] = page_rows
            evidence["page_worker_pods"] = sorted(
                {
                    row["worker_hostname"]
                    for row in page_rows
                    if isinstance(row, dict)
                    and row.get("status") in {"ok", "fallback", "image_only"}
                    and isinstance(row.get("worker_hostname"), str)
                    and row.get("worker_hostname")
                }
            )
            if not runtime_images.get("worker"):
                broker.capture_worker_runtime_identity(
                    kubeconfig, namespace, evidence["release"], runtime_images
                )
            return final_state
        if not runtime_images.get("worker"):
            broker.capture_worker_runtime_identity(
                kubeconfig, namespace, evidence["release"], runtime_images
            )
        time.sleep(1)
    return final_state


def safe_submit_record(value: object) -> dict:
    row = value if isinstance(value, dict) else {}
    job_id = row.get("job_id")
    digest = row.get("source_sha256")
    idempotency_digest = row.get("idempotency_key_sha256")
    status = row.get("status")
    return {
        "http_status": row.get("http_status")
        if type(row.get("http_status")) is int
        else None,
        "job_id": job_id
        if isinstance(job_id, str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            job_id,
        )
        else None,
        "status": status
        if isinstance(status, str)
        and status in {"queued", "processing", "completed", "failed", "cancelled"}
        else "unknown",
        "idempotency_key_sha256": idempotency_digest
        if isinstance(idempotency_digest, str)
        and re.fullmatch(r"[a-f0-9]{64}", idempotency_digest)
        else None,
        "source_sha256": digest
        if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
        else None,
        "source_size_bytes": _bounded_count(row.get("source_size_bytes")),
        "submitted_at": row.get("submitted_at")
        if isinstance(row.get("submitted_at"), str) and len(row["submitted_at"]) <= 64
        else None,
    }


def kubectl_python(
    kubeconfig: Path, namespace: str, deployment: str, source: str
) -> dict:
    return safe_fault_matrix(
        _exec_coordinator_python(kubeconfig, namespace, deployment, source)
    )


def cleanup_tenant_source(tenant_id: str) -> str:
    """Build a cleanup command for this runner's generated tenant only."""
    return f"""
import io, json, os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from django.core.management import call_command
captured = io.StringIO()
try:
    call_command('purge_tenant', '--tenant-id='+{tenant_id!r}, '--confirm',
        '--include-custody', stdout=captured, stderr=captured)
    print(json.dumps({{'ok': True, 'result': 'purge_command_succeeded'}}))
except Exception as exc:
    print(json.dumps({{'ok': False, 'result': 'purge_command_failed',
        'failure_type': type(exc).__name__}}))
"""


def cleanup_tenant(
    kubeconfig: Path, namespace: str, deployment: str, tenant_id: str
) -> dict[str, bool | str]:
    raw = _exec_coordinator_python(
        kubeconfig, namespace, deployment, cleanup_tenant_source(tenant_id)
    )
    ok = (
        isinstance(raw, dict)
        and raw.get("ok") is True
        and raw.get("result") == "purge_command_succeeded"
    )
    return {
        "attempted": True,
        "ok": ok,
        "result": "purge_command_succeeded" if ok else "purge_command_failed",
    }


def safe_cleanup_tenant(
    kubeconfig: Path, namespace: str, deployment: str, tenant_id: str
) -> dict[str, bool | str]:
    try:
        return cleanup_tenant(kubeconfig, namespace, deployment, tenant_id)
    except Exception:
        return {"attempted": True, "ok": False, "result": "purge_command_failed"}


def retention_fault_matrix_source(tenant_id: str, job_id: str) -> str:
    return f"""
import contextlib, hashlib, io, json, os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from unittest.mock import patch
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db.models.query import QuerySet
from django.utils import timezone
from jobs.models import Job, PageResult, PiiEntity, CustodyEvent, TaskDispatchOutbox
from jobs.outbox import enqueue_task_dispatch
from jobs.rls import control_plane_rls_context
from jobs.tasks import _get_backend_for_job as real_get_backend
from jobs.management.commands import purge_tenant as purge_module

TENANT = {tenant_id!r}
JOB_ID = {job_id!r}
PURGE_DOC = purge_module._tenant_purge_document_id(TENANT)
INJECTED_STAGE = "retention-canary-pending"

def snapshot():
    with control_plane_rls_context():
        job = Job.objects.filter(job_id=JOB_ID, tenant_id=TENANT).first()
        dispatches = list(TaskDispatchOutbox.objects.filter(job_id=JOB_ID).order_by('stage').values(
            'stage','generation','status','attempts','last_error'))
        purge_events = list(CustodyEvent.objects.filter(tenant_id=TENANT, document_id=PURGE_DOC)
            .order_by('timestamp','id').values_list('event_type', flat=True))
        result = {{
            'tenant_jobs': Job.objects.filter(tenant_id=TENANT).count(),
            'pages': PageResult.objects.filter(job__tenant_id=TENANT).count(),
            'pii': PiiEntity.objects.filter(job__tenant_id=TENANT).count(),
            'tenant_dispatch_rows': TaskDispatchOutbox.objects.filter(job__tenant_id=TENANT).count(),
            'tenant_job_custody_rows': CustodyEvent.objects.filter(job__tenant_id=TENANT).count(),
            'purge_events': purge_events,
            'job': None if job is None else {{'status': job.status,
                'dispatch_generation': job.dispatch_generation,
                'pages_completed': job.pages_completed, 'total_pages': job.total_pages}},
            'dispatches': dispatches,
        }}
        # Keep storage authority after SQL deletion: absence of a Job row
        # does not establish that its object prefix was removed.
        result['object_keys'] = storage_backend.list_objects(f'jobs/{{JOB_ID}}/')
        return result

def run_command(*extra):
    captured = io.StringIO()
    try:
        call_command('purge_tenant', '--tenant-id='+TENANT, *extra,
            stdout=captured, stderr=captured)
        return {{'raised': False, 'output': captured.getvalue()}}
    except Exception as exc:
        return {{'raised': True, 'exception': type(exc).__name__,
            'message': str(exc), 'output': captured.getvalue()}}

with control_plane_rls_context():
    job = Job.objects.get(job_id=JOB_ID, tenant_id=TENANT)
    storage_backend = real_get_backend(job)
    pending = enqueue_task_dispatch(job=job, stage=INJECTED_STAGE,
        generation=str(job.dispatch_generation), task_name='jobs.tasks.process_document',
        args=[str(job.job_id)], kwargs={{'tenant_id': TENANT}}, queue='coordinator')
    if TaskDispatchOutbox.objects.filter(
        job=job, stage=INJECTED_STAGE, generation=str(job.dispatch_generation)
    ).count() != 1:
        raise RuntimeError('retention dispatch contract was not unique')
    initial_generation = int(job.dispatch_generation)

dry_run = run_command('--dry-run', '--include-custody')
before = snapshot()

def inject_after_query_update(match):
    original = QuerySet.update
    def replacement(queryset, **kwargs):
        result = original(queryset, **kwargs)
        if match(kwargs):
            raise RuntimeError('injected_after_' + ('generation_fence' if 'last_failure_reason' in kwargs else 'outbox_fence'))
        return result
    return patch.object(QuerySet, 'update', replacement)

faults = []
for name, match in [
    ('generation_fence', lambda values: values.get('last_failure_reason') == 'tenant_purge_fence'),
    ('outbox_fence', lambda values: values.get('last_error') == 'tenant purge fenced dispatch'),
]:
    with inject_after_query_update(match):
        attempt = run_command('--confirm', '--include-custody')
    after = snapshot()
    faults.append({{'boundary': name, 'attempt': attempt, 'snapshot': after,
        'state_unchanged': after == before}})

real_record = purge_module._record_tenant_purge_custody_event
def fail_started_custody(*args, **kwargs):
    result = real_record(*args, **kwargs)
    if kwargs.get('event_type') == 'tenant_purge_started':
        raise RuntimeError('injected_after_start_custody_insert')
    return result
with patch.object(purge_module, '_record_tenant_purge_custody_event', fail_started_custody):
    attempt = run_command('--confirm', '--include-custody')
after_custody = snapshot()
faults.append({{'boundary': 'start_custody_commit', 'attempt': attempt,
    'snapshot': after_custody, 'state_unchanged': after_custody == before}})

# Delete exactly one object, then fail the S3 boundary. The command must leave
# the fenced SQL state intact and retryable until remaining objects are removed.
storage_fault = {{'fired': False, 'deleted_before_failure': 0}}
original_get = real_get_backend
def get_backend_with_partial_delete(candidate):
    backend = original_get(candidate)
    if str(candidate.job_id) == JOB_ID and not storage_fault['fired']:
        original_delete = backend.delete_many
        storage_fault['backend'] = backend
        storage_fault['original_delete'] = original_delete
        def delete_one_then_fail(keys):
            keys = list(keys)
            if not keys:
                return 0
            storage_fault['deleted_before_failure'] = original_delete(keys[:1])
            storage_fault['fired'] = True
            raise RuntimeError('injected_after_partial_minio_delete')
        backend.delete_many = delete_one_then_fail
    return backend
with patch('jobs.tasks._get_backend_for_job', side_effect=get_backend_with_partial_delete):
    partial_attempt = run_command('--confirm', '--include-custody')
if storage_fault.get('backend') is not None:
    storage_fault['backend'].delete_many = storage_fault['original_delete']
partial_snapshot = snapshot()
faults.append({{'boundary': 'partial_minio_object_delete', 'attempt': partial_attempt,
    'snapshot': partial_snapshot, 'injection': {{
        'fired': storage_fault['fired'],
        'deleted_before_failure': storage_fault['deleted_before_failure'],
    }}}})

# Capture the second fence at the retry's actual deletion boundary, then let
# the unpatched production command remove the remainder and commit SQL deletes.
retry_boundary = {{}}
def get_backend_with_retry_observer(candidate):
    backend = original_get(candidate)
    if str(candidate.job_id) == JOB_ID:
        original_delete = backend.delete_many
        def observe_then_delete(keys):
            keys = list(keys)
            retry_boundary['job_state'] = snapshot()['job']
            retry_boundary['dispatches'] = snapshot()['dispatches']
            retry_boundary['purge_events'] = snapshot()['purge_events']
            retry_boundary['objects_before_delete'] = len(snapshot()['object_keys'])
            retry_boundary['delete_count'] = original_delete(keys)
            return retry_boundary['delete_count']
        backend.delete_many = observe_then_delete
    return backend
with patch('jobs.tasks._get_backend_for_job', side_effect=get_backend_with_retry_observer):
    retry_attempt = run_command('--confirm', '--include-custody')
final_snapshot = snapshot()

assertions = {{
    'dry_run_reports_one_job_and_two_pages': 'Jobs: 1' in dry_run['output'] and 'Pages: 2' in dry_run['output'],
    'generation_fence_failure_rolls_back': faults[0]['attempt']['raised'] and faults[0]['state_unchanged'],
    'outbox_fence_failure_rolls_back': faults[1]['attempt']['raised'] and faults[1]['state_unchanged'],
    'custody_insert_failure_rolls_back_fence_and_outbox': faults[2]['attempt']['raised'] and faults[2]['state_unchanged'],
    'partial_minio_delete_leaves_fenced_retryable_job': storage_fault['fired'] and partial_attempt['raised'] and partial_snapshot['job']['status'] == 'cancelled' and partial_snapshot['job']['dispatch_generation'] == initial_generation + 1 and len(partial_snapshot['object_keys']) == len(before['object_keys']) - 1,
    'pending_outbox_is_dead_before_storage_delete': any(row['stage'] == INJECTED_STAGE and row['status'] == 'dead' for row in partial_snapshot['dispatches']),
    'start_custody_precedes_minio_delete': 'tenant_purge_started' in partial_snapshot['purge_events'],
    'retry_reestablishes_monotonic_fence_and_removes_remaining_objects': retry_boundary.get('job_state',{{}}).get('status') == 'cancelled' and retry_boundary.get('job_state',{{}}).get('dispatch_generation') == initial_generation + 2 and retry_boundary.get('objects_before_delete',0) > 0 and retry_attempt['raised'] is False,
    'final_tenant_rows_and_minio_prefix_removed': final_snapshot['tenant_jobs'] == 0 and final_snapshot['pages'] == 0 and final_snapshot['tenant_dispatch_rows'] == 0 and final_snapshot['tenant_job_custody_rows'] == 0 and final_snapshot['object_keys'] == [],
    'purge_audit_events_retained': 'tenant_purge_started' in final_snapshot['purge_events'] and 'tenant_purge_job_storage_deleted' in final_snapshot['purge_events'] and 'tenant_purged' in final_snapshot['purge_events'],
}}
print(json.dumps({{'ok': all(assertions.values()), 'initial_generation': initial_generation,
    'pending_dispatch_id': str(pending.dispatch_id), 'dry_run': dry_run, 'before': before,
    'faults': faults, 'partial_snapshot': partial_snapshot,
    'retry_boundary': retry_boundary, 'retry_attempt': retry_attempt,
    'final_snapshot': final_snapshot, 'assertions': assertions}}, sort_keys=True))
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.api_url = require_loopback_http_url(args.api_url, label="--api-url")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    api_key = os.environ.get("OCR_COORDINATOR_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OCR_COORDINATOR_API_KEY is required")
    if not 30 <= args.timeout_sec <= 3600:
        raise SystemExit("--timeout-sec must be between 30 and 3600")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?", args.namespace):
        raise SystemExit("--namespace must be a DNS label")
    if len(args.release) > 53 or not re.fullmatch(
        r"[a-z0-9](?:[-a-z0-9]{0,51}[a-z0-9])?", args.release
    ):
        raise SystemExit("--release must be a DNS label")
    kubeconfig = args.kubeconfig.expanduser().resolve(strict=True)
    fixture = args.fixture.expanduser().resolve(strict=True)
    if not fixture.is_relative_to(REPO_ROOT):
        raise SystemExit("fixture must be inside this checkout")
    try:
        out = require_fresh_directory(args.out, root=REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    context = broker.command([*broker.kube(kubeconfig), "config", "current-context"])
    if not context["ok"]:
        raise SystemExit("could not inspect the supplied kube context")
    nodes = broker.kubectl_json(kubeconfig, ["get", "nodes"])
    names = [
        str((row.get("metadata") or {}).get("name", ""))
        for row in nodes.get("items", [])
    ]
    try:
        require_isolated_kind_identity(context["stdout"].strip(), args.cluster, names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for node_name in names:
        pg8.verify_kind_node(args.cluster, node_name)

    tenant_id = f"sparks-minio-retention-{uuid.uuid4().hex[:12]}"
    coordinator = f"{args.release}-coordinator"
    evidence = {
        "schema_version": 3,
        "lane": "CT-TENANT-RETENTION MinIO fault matrix",
        "claim_tier": "single-node-kind-lab",
        "started_at": pg8.now_iso(),
        "cluster": args.cluster,
        "namespace": args.namespace,
        "release": args.release,
        "tenant_id": tenant_id,
        "fixture": fixture.relative_to(REPO_ROOT).as_posix(),
        "runtime_images": None,
        "limits": [
            "Local PostgreSQL/RLS and local MinIO only; no real S3/IAM semantics are established.",
            "This is a one-node kind run; no production retention atomicity or replicated restore claim.",
        ],
    }
    path = out / "minio-retention-live.json"
    submission_attempted = False
    try:
        health = broker.submit_api(args.api_url, api_key)
        if health.get("http_status") != 200:
            raise RuntimeError("coordinator_health_failed")
        runtime_images = broker.current_source_runtime_images(
            kubeconfig, args.cluster, args.namespace, args.release
        )
        evidence["runtime_images"] = runtime_images
        submission_attempted = True
        submitted = pg8.submit_job(args.api_url, api_key, tenant_id, fixture)
        if submitted.get("http_status") != 201 or not submitted.get("job_id"):
            raise RuntimeError("coordinator_submit_failed")
        job_id = str(submitted["job_id"])
        safe_submit = safe_submit_record(submitted)
        if (
            safe_submit["http_status"] != 201
            or safe_submit["job_id"] != job_id
            or safe_submit["source_sha256"] is None
        ):
            raise RuntimeError("coordinator_submit_result_invalid")
        evidence["job_id"] = job_id
        evidence["submit"] = safe_submit
        pages_result = broker.command(["pdfinfo", str(fixture)], timeout=15)
        match = re.search(
            r"^Pages:\s+(\d+)\s*$", pages_result.get("stdout", ""), re.MULTILINE
        )
        expected_pages = (
            int(match.group(1)) if pages_result.get("ok") and match else None
        )
        if expected_pages is None:
            raise RuntimeError("fixture_page_count_unavailable")

        final_state = observe_job_and_capture_worker(
            kubeconfig,
            args.namespace,
            coordinator,
            job_id,
            runtime_images,
            evidence,
            args.timeout_sec,
        )
        if not final_state or final_state["job"]["status"] != "completed":
            raise RuntimeError("job_did_not_complete")
        if not runtime_images.get("worker"):
            raise RuntimeError(
                "completed retention job lacks a Ready source-bound worker identity"
            )
        worker_identity = runtime_images.get("worker")
        if not isinstance(worker_identity, dict) or evidence.get(
            "page_worker_pods"
        ) != [worker_identity.get("pod")]:
            raise RuntimeError(
                "completed retention pages do not match the captured source-bound worker"
            )
        api_final = pg8.api_status(args.api_url, api_key, tenant_id, job_id)
        raw_artifacts = broker.active_artifact_check(
            kubeconfig, args.namespace, coordinator, job_id
        )
        artifacts = safe_active_artifacts(raw_artifacts)
        if (
            final_state["job"]["total_pages"] != expected_pages
            or final_state["job"]["pages_completed"] != expected_pages
            or final_state["job"]["pages_failed"] != 0
            or api_final.get("http_status") != 200
            or not artifacts["ok"]
        ):
            raise RuntimeError("live_minio_job_or_output_preflight_failed")

        remote = kubectl_python(
            kubeconfig,
            args.namespace,
            coordinator,
            retention_fault_matrix_source(tenant_id, job_id),
        )
        evidence.update(
            {
                "started_at": submitted.get("submitted_at") or evidence["started_at"],
                "job_id": job_id,
                "source_sha256": safe_submit["source_sha256"],
                "submit": safe_submit,
                "job_before_purge": safe_job_before_purge(final_state["job"]),
                "api_status_before_purge": api_final,
                "active_minio_artifacts_before_purge": artifacts,
                "fault_matrix": remote,
            }
        )
        evidence["ok"] = bool(remote.get("ok"))
        if not evidence["ok"]:
            evidence["cleanup"] = safe_cleanup_tenant(
                kubeconfig, args.namespace, coordinator, tenant_id
            )
        evidence["failure_type"] = None if evidence["ok"] else "fault_matrix_failed"
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        evidence["ok"] = False
        evidence["failure_type"] = type(exc).__name__
        evidence["failure_reason"] = safe_failure_reason(exc)
        evidence["cleanup"] = (
            safe_cleanup_tenant(kubeconfig, args.namespace, coordinator, tenant_id)
            if submission_attempted
            else {"attempted": False, "ok": True, "result": "not_needed"}
        )

    evidence["status"] = "PASS" if evidence["ok"] else "FAIL"
    evidence["finished_at"] = pg8.now_iso()
    write_json_no_replace(path, evidence)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "job_id": evidence.get("job_id"),
                "tenant_id": tenant_id,
                "fault_matrix": evidence.get("fault_matrix", {}).get("assertions", {}),
                "cleanup": evidence.get("cleanup"),
                "evidence": str(path),
            }
        )
    )
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
