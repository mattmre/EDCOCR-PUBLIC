#!/usr/bin/env python3
"""Inject one live OCR worker loss after a durable page result on isolated kind.

The script submits through the Django coordinator API, waits until one page is
committed and a later page is actively leased, then stops the exact
KEDA-managed worker container through the isolated kind node's CRI. It retains
the before/after database state and container logs. It never creates or
changes a Kubernetes context; ``--kubeconfig`` and ``--cluster`` must identify
the exact kind cluster to use.

Required environment: ``OCR_COORDINATOR_API_KEY`` (local lab key).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from runtime_image_identity import verified_pod_container_identity  # noqa: E402
from runtime_source_fingerprint import fingerprint_for_profile  # noqa: E402
from vtm_canary_safety import (  # noqa: E402
    open_no_redirect,
    require_fresh_directory,
    require_isolated_kind_identity,
    require_loopback_http_url,
    require_uuid_job_id,
    write_bytes_no_replace,
    write_json_no_replace,
)

SUCCESS_PAGE_STATES = ("ok", "fallback", "image_only")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def page_task_execution_success_logged(
    log_text: str, *, task_id: str, page_num: int
) -> bool:
    """Require a worker success log for this exact Celery task and page."""
    if not task_id or type(page_num) is not int or page_num < 1:
        return False
    match = re.search(
        rf"\bTask jobs\.tasks\.process_page\[{re.escape(task_id)}\] succeeded\b([^\r\n]*)",
        log_text,
    )
    return bool(
        match
        and re.search(
            rf"['\"]page_num['\"]\s*:\s*{page_num}(?=[,}}\s])",
            match.group(1),
        )
    )


def run(cmd: list[str], timeout: int = 30, *, input_text: str | None = None) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cmd": cmd, "ok": False, "error": repr(exc)}


def kubectl_base(kubeconfig: Path) -> list[str]:
    return ["kubectl", "--kubeconfig", str(kubeconfig)]


def kubectl_json(kubeconfig: Path, args: list[str], timeout: int = 30) -> Any:
    result = run([*kubectl_base(kubeconfig), *args, "-o", "json"], timeout=timeout)
    if not result["ok"]:
        raise RuntimeError("kubectl request failed")
    try:
        return json.loads(result["stdout"] or "null")
    except json.JSONDecodeError as exc:
        raise RuntimeError("kubectl returned invalid JSON") from exc


def django_state(
    kubeconfig: Path, namespace: str, coordinator_deployment: str, job_id: str
) -> dict:
    source = f"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from jobs.models import Job, PageResult, TaskDispatchOutbox, CustodyEvent
j = Job.objects.get(job_id={job_id!r})
pages_qs = PageResult.objects.filter(job=j).order_by('page_num')
pages = list(pages_qs.values('page_num', 'status', 'celery_task_id', 'worker_hostname'))
dispatches = list(TaskDispatchOutbox.objects.filter(job=j).order_by('created_at').values(
  'stage', 'generation', 'broker_task_id', 'status', 'attempts'))
custody = [{{'event_type': e.event_type, 'event_hash': e.event_hash,
  'timestamp': e.timestamp.isoformat()}} for e in
  CustodyEvent.objects.filter(job=j).order_by('timestamp', 'id')]
settings = j.settings_json if isinstance(j.settings_json, dict) else {{}}
print(json.dumps({{
  'job': {{'job_id': str(j.job_id), 'status': j.status, 'total_pages': j.total_pages,
    'pages_completed': j.pages_completed, 'pages_failed': j.pages_failed,
    'source_hash': j.source_hash,
    'dispatch_generation': j.dispatch_generation,
    'settings_json': {{'_active_output_generation': settings.get('_active_output_generation')}}}},
  'pages': pages,
  'dispatches': dispatches,
  'custody': custody
}}))
"""
    command = [
        *kubectl_base(kubeconfig),
        "exec",
        "-n",
        namespace,
        f"deployment/{coordinator_deployment}",
        "-c",
        "coordinator",
        "--",
        "python3",
        "-c",
        source,
    ]
    result = run(command, timeout=30)
    if not result["ok"]:
        raise RuntimeError("Django state query failed")
    lines = [
        line for line in result["stdout"].splitlines() if line.strip().startswith("{")
    ]
    if not lines:
        raise RuntimeError("Django state query returned no JSON")
    value = safe_django_state(json.loads(lines[-1]))
    successes = [p for p in value["pages"] if p["status"] in SUCCESS_PAGE_STATES]
    processing = [p for p in value["pages"] if p["status"] == "processing"]
    value["observed"] = {
        "at": now_iso(),
        "successful_page_count": len(successes),
        "max_successful_page": max((p["page_num"] for p in successes), default=0),
        "processing_pages": processing,
    }
    return value


def safe_django_state(value: Any) -> dict[str, Any]:
    """Retain only fields required to validate PG-8 completion and replay."""
    if not isinstance(value, dict):
        return {}

    raw_job = value.get("job")
    safe_job: dict[str, Any] = {}
    if isinstance(raw_job, dict):
        for key in ("job_id", "status", "source_hash"):
            item = raw_job.get(key)
            if isinstance(item, str) and len(item) <= 128:
                safe_job[key] = item
        for key in (
            "total_pages",
            "pages_completed",
            "pages_failed",
            "dispatch_generation",
        ):
            item = raw_job.get(key)
            if type(item) is int and 0 <= item <= 1_000_000:
                safe_job[key] = item

        raw_settings = raw_job.get("settings_json")
        raw_pointer = (
            raw_settings.get("_active_output_generation")
            if isinstance(raw_settings, dict)
            else None
        )
        if isinstance(raw_pointer, dict):
            pointer: dict[str, Any] = {}
            raw_artifacts = raw_pointer.get("artifacts")
            if isinstance(raw_artifacts, dict):
                artifacts: dict[str, dict[str, str]] = {}
                for role in ("pdf", "text"):
                    raw_artifact = raw_artifacts.get(role)
                    if not isinstance(raw_artifact, dict):
                        continue
                    artifact = {
                        key: item
                        for key in ("sha256", "generation_id", "relative_path")
                        if isinstance((item := raw_artifact.get(key)), str)
                        and len(item) <= 512
                    }
                    artifacts[role] = artifact
                pointer["artifacts"] = artifacts
            safe_job["settings_json"] = {"_active_output_generation": pointer}

    pages = []
    raw_pages = value.get("pages")
    if isinstance(raw_pages, list):
        for page in raw_pages:
            if not isinstance(page, dict):
                continue
            safe_page: dict[str, Any] = {}
            page_num = page.get("page_num")
            if type(page_num) is int and 0 <= page_num <= 1_000_000:
                safe_page["page_num"] = page_num
            for key in ("status", "celery_task_id", "worker_hostname"):
                item = page.get(key)
                if isinstance(item, str) and len(item) <= 256:
                    safe_page[key] = item
            pages.append(safe_page)

    dispatches = []
    raw_dispatches = value.get("dispatches")
    if isinstance(raw_dispatches, list):
        for dispatch in raw_dispatches:
            if not isinstance(dispatch, dict):
                continue
            safe_dispatch: dict[str, str | int] = {}
            for key in ("stage", "broker_task_id", "status"):
                item = dispatch.get(key)
                if isinstance(item, str) and len(item) <= 256:
                    safe_dispatch[key] = item
            attempts = dispatch.get("attempts")
            if type(attempts) is int and 0 <= attempts <= 1_000_000:
                safe_dispatch["attempts"] = attempts
            generation = dispatch.get("generation")
            if (
                generation is None
                or (type(generation) is int and 0 <= generation <= 1_000_000)
                or (
                    isinstance(generation, str)
                    and re.fullmatch(r"[0-9]{1,64}", generation) is not None
                )
            ):
                safe_dispatch["generation"] = generation
            dispatches.append(safe_dispatch)

    custody = []
    raw_custody = value.get("custody")
    if isinstance(raw_custody, list):
        for event in raw_custody:
            if not isinstance(event, dict):
                continue
            safe_event: dict[str, str] = {}
            for key in ("event_type", "event_hash", "timestamp"):
                item = event.get(key)
                if isinstance(item, str) and len(item) <= 256:
                    safe_event[key] = item
            custody.append(safe_event)

    return {
        "job": safe_job,
        "pages": pages,
        "dispatches": dispatches,
        "custody": custody,
    }


def django_observation(
    kubeconfig: Path, namespace: str, coordinator_deployment: str, job_id: str
) -> dict:
    """Read only the counters and leases needed by the high-frequency watcher."""
    source = f"""
import os, django, json
from django.db.models import Max
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from jobs.models import Job, PageResult
j = Job.objects.get(job_id={job_id!r})
success = PageResult.objects.filter(job=j, status__in={SUCCESS_PAGE_STATES!r})
processing = list(PageResult.objects.filter(job=j, status='processing').values(
  'page_num', 'status', 'celery_task_id', 'worker_hostname'))
print(json.dumps({{
  'job': {{'job_id': str(j.job_id), 'status': j.status, 'total_pages': j.total_pages,
    'pages_completed': j.pages_completed, 'pages_failed': j.pages_failed,
    'source_hash': j.source_hash, 'dispatch_generation': j.dispatch_generation}},
  'successful_page_count': success.count(),
  'max_successful_page': success.aggregate(value=Max('page_num'))['value'] or 0,
  'processing_pages': processing
}}))
"""
    command = [
        *kubectl_base(kubeconfig),
        "exec",
        "-n",
        namespace,
        f"deployment/{coordinator_deployment}",
        "-c",
        "coordinator",
        "--",
        "python3",
        "-c",
        source,
    ]
    result = run(command, timeout=30)
    if not result["ok"]:
        raise RuntimeError("Django observation query failed")
    lines = [
        line for line in result["stdout"].splitlines() if line.strip().startswith("{")
    ]
    if not lines:
        raise RuntimeError("Django observation query returned no JSON")
    value = json.loads(lines[-1])
    value["observed_at"] = now_iso()
    return value


def start_page_boundary_watcher(
    kubeconfig: Path, namespace: str, coordinator_deployment: str
) -> subprocess.Popen[str]:
    """Warm Django before submit, then watch the first durable/active overlap."""
    source = f"""
import datetime, json, os, sys, time, django
from django.db.models import Max
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from jobs.models import Job, PageResult
print('PG8_WATCHER_READY', flush=True)
job_id = sys.stdin.readline().strip()
success_states = {SUCCESS_PAGE_STATES!r}
while job_id:
    job = Job.objects.get(job_id=job_id)
    success = PageResult.objects.filter(job=job, status__in=success_states)
    processing = list(PageResult.objects.filter(job=job, status='processing').values(
        'page_num', 'status', 'celery_task_id', 'worker_hostname'))
    count = success.count()
    if count > 0 and processing:
        print(json.dumps({{
            'job': {{'job_id': str(job.job_id), 'status': job.status,
                'total_pages': job.total_pages, 'pages_completed': job.pages_completed,
                'pages_failed': job.pages_failed, 'source_hash': job.source_hash,
                'dispatch_generation': job.dispatch_generation}},
            'successful_page_count': count,
            'max_successful_page': success.aggregate(value=Max('page_num'))['value'] or 0,
            'processing_pages': processing,
            'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }}), flush=True)
        break
    if job.status in {{'completed', 'failed', 'cancelled'}}:
        print(json.dumps({{'job': {{'job_id': str(job.job_id), 'status': job.status}},
            'successful_page_count': count, 'processing_pages': processing,
            'terminal': True}}), flush=True)
        break
    time.sleep(0.02)
"""
    watcher = subprocess.Popen(
        [
            *kubectl_base(kubeconfig),
            "exec",
            "-i",
            "-n",
            namespace,
            f"deployment/{coordinator_deployment}",
            "-c",
            "coordinator",
            "--",
            "python3",
            "-u",
            "-c",
            source,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if watcher.stdout is None:
        watcher.terminate()
        raise RuntimeError("PG-8 database watcher has no stdout pipe")
    ready, _, _ = select.select([watcher.stdout], [], [], 60)
    if not ready or watcher.stdout.readline().strip() != "PG8_WATCHER_READY":
        watcher.terminate()
        if watcher.stderr:
            watcher.stderr.read()
        raise RuntimeError("PG-8 database watcher failed to start")
    return watcher


def await_page_boundary(
    watcher: subprocess.Popen[str], job_id: str, timeout_sec: int
) -> dict[str, Any]:
    if watcher.stdin is None or watcher.stdout is None:
        raise RuntimeError("PG-8 database watcher pipes are unavailable")
    watcher.stdin.write(job_id + "\n")
    watcher.stdin.flush()
    watcher.stdin.close()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        ready, _, _ = select.select([watcher.stdout], [], [], 1.0)
        if ready:
            line = watcher.stdout.readline()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
                raise RuntimeError("PG-8 watcher returned non-object JSON")
        if watcher.poll() is not None:
            if watcher.stderr:
                watcher.stderr.read()
            raise RuntimeError("PG-8 database watcher exited early")
    raise TimeoutError("timed out waiting for durable-page/active-page overlap")


def api_request(
    url: str,
    api_key: str,
    tenant_id: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    request_headers = {
        "X-API-Key": api_key,
        "X-Tenant-ID": tenant_id,
        **(headers or {}),
    }
    request = urllib.request.Request(
        url, data=data, method=method, headers=request_headers
    )
    try:
        with open_no_redirect(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def submit_job(api_url: str, api_key: str, tenant_id: str, fixture: Path) -> dict:
    boundary = f"ocr-local-{uuid.uuid4().hex}"
    content = fixture.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; filename="{fixture.name}"\r\n'
                "Content-Type: application/pdf\r\n\r\n"
            ).encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    idempotency_key = f"pg8-worker-recovery-{uuid.uuid4()}"
    status, raw = api_request(
        f"{api_url.rstrip('/')}/api/v1/jobs",
        api_key,
        tenant_id,
        method="POST",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": idempotency_key,
        },
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        payload = {}
    if status != 201 or not payload.get("job_id"):
        raise RuntimeError(f"coordinator API submit failed: HTTP {status}")
    try:
        job_id = require_uuid_job_id(payload["job_id"])
    except ValueError as exc:
        raise RuntimeError("coordinator API returned an invalid job ID") from exc
    return {
        "http_status": status,
        "idempotency_key_sha256": hashlib.sha256(idempotency_key.encode()).hexdigest(),
        "job_id": job_id,
        "status": payload.get("status"),
        "source_file": fixture.name,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_size_bytes": len(content),
        "submitted_at": now_iso(),
    }


def api_status(api_url: str, api_key: str, tenant_id: str, job_id: str) -> dict:
    status, raw = api_request(
        f"{api_url.rstrip('/')}/api/v1/jobs/{job_id}", api_key, tenant_id
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        payload = {}
    return {
        "http_status": status,
        "payload": safe_api_status_payload(payload),
        "observed_at": now_iso(),
    }


def safe_api_status_payload(payload: Any) -> dict[str, str | int]:
    """Keep only bounded completion fields needed by the PG-8 validator."""
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str | int] = {}
    for key in ("job_id", "status", "source_file_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and len(value) <= 128:
            result[key] = value
    for key in ("total_pages", "pages_completed", "pages_failed"):
        value = payload.get(key)
        if type(value) is int and 0 <= value <= 1_000_000:
            result[key] = value
    return result


def failure_record(exc: Exception, *, stage: str = "unknown") -> dict[str, str | bool]:
    """Record failure class without persisting arbitrary exception text."""
    return {
        "status": "FAIL",
        "ok": False,
        "error_type": type(exc).__name__,
        "stage": stage,
        "at": now_iso(),
    }


def running_worker(kubeconfig: Path, namespace: str, selector: str) -> dict | None:
    obj = kubectl_json(
        kubeconfig,
        ["get", "pods", "-n", namespace, "-l", selector],
    )
    for pod in obj.get("items", []) if isinstance(obj, dict) else []:
        status = pod.get("status") or {}
        if status.get("phase") != "Running":
            continue
        if (pod.get("metadata") or {}).get("deletionTimestamp"):
            continue
        if not any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in status.get("conditions", [])
        ):
            continue
        name = (pod.get("metadata") or {}).get("name")
        if name:
            return pod
    return None


def worker_for_hostname(pods: dict, worker_hostname: str) -> dict | None:
    """Select only the ready Pod that owns the active page lease."""
    if not worker_hostname or not isinstance(pods, dict):
        return None
    matches = []
    for pod in pods.get("items", []):
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        name = str(metadata.get("name", ""))
        if (
            name != worker_hostname
            or status.get("phase") != "Running"
            or metadata.get("deletionTimestamp")
        ):
            continue
        pod_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions") or []
        )
        container_ready = container_status(pod, "cpu-ocr-worker").get("ready") is True
        if not pod_ready or not container_ready:
            continue
        matches.append(pod)
    return matches[0] if len(matches) == 1 else None


def running_worker_for_hostname(
    kubeconfig: Path,
    namespace: str,
    selector: str,
    worker_hostname: str,
    *,
    timeout_sec: float = 8.0,
    poll_sec: float = 0.1,
) -> dict | None:
    """Wait briefly for the active lease owner to pass Pod startup readiness."""
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        pods = kubectl_json(
            kubeconfig,
            ["get", "pods", "-n", namespace, "-l", selector],
        )
        pod = worker_for_hostname(pods, worker_hostname)
        if pod is not None:
            return pod
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.01, poll_sec))


def running_component_pod(
    kubeconfig: Path, namespace: str, release: str, component: str
) -> dict | None:
    obj = kubectl_json(
        kubeconfig,
        [
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app.kubernetes.io/instance={release},app.kubernetes.io/component={component}",
        ],
    )
    for pod in obj.get("items", []) if isinstance(obj, dict) else []:
        status = pod.get("status") or {}
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions") or []
        )
        if status.get("phase") == "Running" and ready:
            return pod
    return None


def container_status(pod: dict, container_name: str) -> dict:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    return next((c for c in statuses if c.get("name") == container_name), {})


def verify_kind_node(cluster: str, node_name: str) -> dict[str, str]:
    """Require a Docker node container labeled for this exact kind cluster."""
    if not cluster or not node_name or "/" in node_name:
        raise RuntimeError("refusing CRI access without an exact kind cluster/node")
    inspected = run(["docker", "inspect", node_name], timeout=15)
    if not inspected.get("ok"):
        raise RuntimeError(f"could not verify isolated kind node {node_name!r}")
    try:
        values = json.loads(inspected.get("stdout", ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker returned invalid kind-node inspection JSON") from exc
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise RuntimeError("Docker inspection must identify exactly one kind node")
    value = values[0]
    labels = (value.get("Config") or {}).get("Labels") or {}
    actual_name = str(value.get("Name", "")).lstrip("/")
    actual_cluster = str(labels.get("io.x-k8s.kind.cluster", ""))
    role = str(labels.get("io.x-k8s.kind.role", ""))
    if (
        actual_name != node_name
        or actual_cluster != cluster
        or role
        not in {
            "control-plane",
            "worker",
        }
    ):
        raise RuntimeError(
            "refusing CRI access: Docker node name/cluster/role labels do not "
            f"match the requested kind target ({actual_name!r}, "
            f"{actual_cluster!r}, {role!r})"
        )
    return {"name": actual_name, "cluster": actual_cluster, "role": role}


def stop_worker_container(
    cluster: str,
    node_name: str,
    container_ref: str,
) -> dict:
    """Stop the exact OCR container through its verified isolated kind node."""
    runtime, separator, container_id = container_ref.partition("://")
    if (
        runtime != "containerd"
        or not separator
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
    ):
        raise RuntimeError(
            "refusing CRI stop: expected a containerd runtime and a full "
            "64-character container ID"
        )
    node = verify_kind_node(cluster, node_name)
    stopped = run(
        [
            "docker",
            "exec",
            node_name,
            "crictl",
            "stop",
            "--timeout",
            "0",
            container_id,
        ],
        timeout=30,
    )
    return {
        "ok": stopped.get("ok") is True,
        "method": "stop exact OCR worker container through isolated kind-node CRI",
        "node": node["name"],
        "cluster": node["cluster"],
        "container_id": container_id,
        "exit_code": stopped.get("exit_code"),
    }


def require_successful_worker_loss(stop_result: dict) -> None:
    """Fail closed unless the exact targeted worker container was stopped."""
    if stop_result.get("ok") is not True:
        raise RuntimeError("worker-loss injection command did not succeed")


def write_json(path: Path, value: dict) -> None:
    write_json_no_replace(path, value)


def capture_logs(kubeconfig: Path, namespace: str, pod: str, out: Path) -> dict:
    results = {}
    for label, extra in (("previous", ["--previous"]), ("current", [])):
        result = run(
            [
                *kubectl_base(kubeconfig),
                "logs",
                "-n",
                namespace,
                f"pod/{pod}",
                "-c",
                "cpu-ocr-worker",
                *extra,
            ],
            timeout=40,
        )
        if result["ok"]:
            write_bytes_no_replace(
                out / f"worker-{label}.log", result["stdout"].encode("utf-8")
            )
        results[label] = {
            "ok": result["ok"],
            "exit_code": result.get("exit_code"),
            "path": f"worker-{label}.log" if result["ok"] else None,
        }
    return results


def capture_recovery_log(kubeconfig: Path, namespace: str, pod: str) -> dict:
    result = run(
        [
            *kubectl_base(kubeconfig),
            "logs",
            "-n",
            namespace,
            f"pod/{pod}",
            "-c",
            "cpu-ocr-worker",
        ],
        timeout=40,
    )
    return {
        "ok": result.get("ok") is True,
        "exit_code": result.get("exit_code"),
        "path": None,
        "stdout": result.get("stdout", ""),
    }


def capture_committed_artifacts(
    kubeconfig: Path,
    context: str,
    namespace: str,
    release: str,
    job_id: str,
    source_sha256: str,
    out: Path,
) -> dict[str, Any]:
    """Retain committed PDF, text, and custody objects for strict PG-8 review."""
    from vtm_pg7_keda_ocr_canary import _coordinator_artifacts

    args = argparse.Namespace(
        kubeconfig=kubeconfig,
        context=context,
        namespace=namespace,
        release=release,
    )
    payload = _coordinator_artifacts(args, job_id)
    if (
        payload.get("status") != "completed"
        or payload.get("source_sha256") != source_sha256
    ):
        raise RuntimeError(
            "committed output export does not match the completed PG-8 job"
        )
    files = payload.get("files") or {}
    output_dir = out / "job-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "output_pdf": output_dir / "final.pdf",
        "output_text": output_dir / "final.txt",
        "custody": output_dir / f"{source_sha256[:16]}.custody.jsonl",
    }
    records: dict[str, dict[str, Any]] = {}
    for role, target in targets.items():
        data = files.get(role)
        if not isinstance(data, bytes) or not data:
            raise RuntimeError(f"coordinator output export is missing {role}")
        write_bytes_no_replace(target, data)
        records[role] = {
            "path": target.relative_to(out).as_posix(),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "source_sha256": payload.get("source_sha256"),
        "custody_events": payload.get("custody_events"),
        "files": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, help="Exact kind cluster name")
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--namespace", default="ocr-local")
    parser.add_argument("--release", default="ocr-kind")
    parser.add_argument("--api-url", default="http://127.0.0.1:18089")
    parser.add_argument("--tenant-id", default="sparks-c7-lab")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--poll-sec", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.api_url = require_loopback_http_url(args.api_url, label="--api-url")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    api_key = os.environ.get("OCR_COORDINATOR_API_KEY", "").strip()
    if not api_key:
        print(
            "OCR_COORDINATOR_API_KEY must be set to the local lab key", file=sys.stderr
        )
        return 2
    if not re.fullmatch(r"ocr-closeout-c7(?:-[0-9]{8})?", args.cluster):
        print(
            "--cluster must name the isolated ocr-closeout-c7 kind cluster",
            file=sys.stderr,
        )
        return 2
    if not 30 <= args.timeout_sec <= 3600:
        print("--timeout-sec must be between 30 and 3600", file=sys.stderr)
        return 2
    if not 0.05 <= args.poll_sec <= 10:
        print("--poll-sec must be between 0.05 and 10", file=sys.stderr)
        return 2
    fixture = args.fixture.expanduser().resolve(strict=True)
    kubeconfig = args.kubeconfig.expanduser().resolve(strict=True)
    try:
        args.out = require_fresh_directory(args.out, root=REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_fixture_path = args.out / "input" / fixture.name
    source_fixture_path.parent.mkdir()
    write_bytes_no_replace(source_fixture_path, fixture.read_bytes())
    run_started_at = now_iso()
    failure_stage = "cluster_identity"
    try:
        context = run([*kubectl_base(kubeconfig), "config", "current-context"])
        expected_context = f"kind-{args.cluster}"
        if not context["ok"]:
            raise RuntimeError("could not inspect kube context")
        nodes = kubectl_json(kubeconfig, ["get", "nodes"])
        node_names = [
            str((item.get("metadata") or {}).get("name", ""))
            for item in (nodes.get("items") or [])
        ]
        require_isolated_kind_identity(
            context["stdout"].strip(), args.cluster, node_names
        )
        for node_name in node_names:
            verify_kind_node(args.cluster, node_name)

        failure_stage = "runtime_identity_preflight"
        expected_source_fingerprints = {
            "root-cpu": fingerprint_for_profile(REPO_ROOT, "root-cpu"),
            "coordinator": fingerprint_for_profile(REPO_ROOT, "coordinator"),
            "worker": fingerprint_for_profile(REPO_ROOT, "worker"),
        }
        api_pod = running_component_pod(kubeconfig, args.namespace, args.release, "api")
        coordinator_pod = running_component_pod(
            kubeconfig, args.namespace, args.release, "coordinator"
        )
        if api_pod is None or coordinator_pod is None:
            raise RuntimeError("Ready API and coordinator Pods are required")
        api_runtime = verified_pod_container_identity(
            api_pod,
            "api",
            expected_source_tree_sha256=expected_source_fingerprints["root-cpu"],
            expected_source_profile="root-cpu",
        )
        coordinator_runtime = verified_pod_container_identity(
            coordinator_pod,
            "coordinator",
            expected_source_tree_sha256=expected_source_fingerprints["coordinator"],
            expected_source_profile="coordinator",
        )

        page_count_result = run(["pdfinfo", str(fixture)], timeout=15)
        expected_pages = None
        if page_count_result["ok"]:
            match = re.search(r"^Pages:\s+(\d+)\s*$", page_count_result["stdout"], re.M)
            if match:
                expected_pages = int(match.group(1))
        coordinator_deployment = f"{args.release}-coordinator"
        worker_selector = f"app.kubernetes.io/instance={args.release},app.kubernetes.io/component=cpu-ocr-worker"
        initial_worker = running_worker(kubeconfig, args.namespace, worker_selector)
        initial_worker_status = (
            container_status(initial_worker, "cpu-ocr-worker") if initial_worker else {}
        )
        initial_worker_runtime = None
        if initial_worker and initial_worker_status.get("ready") is True:
            initial_worker_runtime = verified_pod_container_identity(
                initial_worker,
                "cpu-ocr-worker",
                expected_source_tree_sha256=expected_source_fingerprints["worker"],
                expected_source_profile="worker",
            )

        failure_stage = "submit_and_observe_page_boundary"
        watcher = start_page_boundary_watcher(
            kubeconfig, args.namespace, coordinator_deployment
        )
        try:
            submitted = submit_job(args.api_url, api_key, args.tenant_id, fixture)
            if expected_pages is not None:
                submitted["expected_pages"] = expected_pages
            write_json(args.out / "coordinator-submit.json", submitted)
            job_id = submitted["job_id"]
            observation = await_page_boundary(watcher, job_id, args.timeout_sec)
        finally:
            if watcher.poll() is None:
                watcher.terminate()
            try:
                watcher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                watcher.kill()
                watcher.wait(timeout=5)
        deadline = time.monotonic() + args.timeout_sec
        boundary: dict | None = None
        worker_pod_name = ""
        if observation.get("successful_page_count", 0) > 0 and observation.get(
            "processing_pages"
        ):
            active_page = observation["processing_pages"][0]
            active_worker_hostname = str(active_page.get("worker_hostname") or "")
            failure_stage = "resolve_active_worker_pod"
            pod = running_worker_for_hostname(
                kubeconfig,
                args.namespace,
                worker_selector,
                active_worker_hostname,
            )
            if pod is None:
                raise RuntimeError(
                    "the worker owning the active page lease is not uniquely "
                    "identified as a Ready OCR worker Pod"
                )
            worker_pod_name = str((pod.get("metadata") or {}).get("name", ""))
            failure_stage = "capture_active_worker_identity"
            before_status = container_status(pod, "cpu-ocr-worker")
            if not worker_pod_name or not before_status:
                raise RuntimeError(
                    "ready KEDA OCR worker pod/container could not be identified"
                )
            live_container_id = str(before_status.get("containerID", ""))
            initial_container_id = str(initial_worker_status.get("containerID", ""))
            if (
                initial_worker_runtime is not None
                and live_container_id
                and live_container_id == initial_container_id
            ):
                worker_runtime = initial_worker_runtime
            else:
                worker_runtime = verified_pod_container_identity(
                    pod,
                    "cpu-ocr-worker",
                    expected_source_tree_sha256=expected_source_fingerprints["worker"],
                    expected_source_profile="worker",
                )
            successful_before = observation["successful_page_count"]
            old_pod_uid = str((pod.get("metadata") or {}).get("uid") or "")
            old_container_id = live_container_id.partition("://")[2]
            restart_count_before = int(before_status.get("restartCount", 0) or 0)
            boundary = {
                "observed_at": now_iso(),
                "cluster": args.cluster,
                "namespace": args.namespace,
                "job_id": job_id,
                "successful_page_count": successful_before,
                "max_successful_page": observation["max_successful_page"],
                "active_page": active_page,
                "worker_pod": worker_pod_name,
                "worker_pod_uid_before": old_pod_uid,
                "worker_container_id_before": old_container_id,
                "worker_container_restart_count_before": restart_count_before,
                "worker_runtime": worker_runtime,
                "fast_observation": observation,
            }
            failure_stage = "capture_worker_before_kill_log"
            before_kill_logs = run(
                [
                    *kubectl_base(kubeconfig),
                    "logs",
                    "-n",
                    args.namespace,
                    f"pod/{worker_pod_name}",
                    "-c",
                    "cpu-ocr-worker",
                ],
                timeout=40,
            )
            if not before_kill_logs.get("ok"):
                raise RuntimeError(
                    "could not capture the active worker log before loss"
                )
            before_kill_log = str(before_kill_logs.get("stdout", "")).encode("utf-8")
            write_bytes_no_replace(args.out / "worker-before-kill.log", before_kill_log)
            boundary["worker_before_kill_log_sha256"] = hashlib.sha256(
                before_kill_log
            ).hexdigest()
            boundary["worker_before_kill_log_captured_at"] = now_iso()
            boundary["kill_at"] = now_iso()
            worker_node_name = str((pod.get("spec") or {}).get("nodeName", ""))
            failure_stage = "stop_exact_worker_container"
            kill_result = stop_worker_container(
                args.cluster,
                worker_node_name,
                before_status.get("containerID", ""),
            )
            boundary["kill_command"] = {
                "issued_at": now_iso(),
                "exit_code": kill_result.get("exit_code"),
                "ok": kill_result.get("ok"),
                "method": kill_result.get("method"),
                "node": kill_result.get("node"),
                "container_id": kill_result.get("container_id"),
            }
            write_json(args.out / "worker-kill-boundary.json", boundary)
            require_successful_worker_loss(kill_result)

        if boundary is None:
            status = api_status(args.api_url, api_key, args.tenant_id, job_id)
            write_json(args.out / "coordinator-status-no-kill.json", status)
            write_json(
                args.out / "recovery-result.json",
                {
                    "status": "NO_INJECTION",
                    "ok": False,
                    "reason": "job reached terminal state or no durable-page/active-page overlap was observed",
                    "job_id": job_id,
                    "source_sha256": submitted["source_sha256"],
                },
            )
            print(json.dumps({"ok": False, "status": "NO_INJECTION", "job_id": job_id}))
            return 1

        failure_stage = "wait_for_worker_replacement"
        restart_deadline = time.monotonic() + 120
        restarted = False
        restart_observations = []
        recovery_worker_pod_name = ""
        recovery_worker_pod_uid = ""
        recovery_worker_container_id = ""
        recovery_worker_restart_count = 0
        recovery_worker_runtime: dict[str, Any] | None = None
        while time.monotonic() < restart_deadline:
            current = running_worker(kubeconfig, args.namespace, worker_selector)
            current_name = (
                str((current.get("metadata") or {}).get("name", "")) if current else ""
            )
            current_uid = (
                str((current.get("metadata") or {}).get("uid", "")) if current else ""
            )
            current_status = (
                container_status(current, "cpu-ocr-worker") if current else {}
            )
            restarts = int(current_status.get("restartCount", 0) or 0)
            restart_observations.append(
                {
                    "at": now_iso(),
                    "pod": current_name,
                    "pod_uid": current_uid,
                    "container_id": current_status.get("containerID"),
                    "phase": (current.get("status") or {}).get("phase")
                    if current
                    else None,
                    "restart_count": restarts,
                    "container_state": current_status.get("state"),
                    "last_termination_state": current_status.get("lastState"),
                }
            )
            if (
                current
                and current_uid
                and current_status.get("ready") is True
                and (current_uid != old_pod_uid or restarts > restart_count_before)
            ):
                restarted = True
                recovery_worker_pod_name = current_name
                recovery_worker_pod_uid = current_uid
                recovery_worker_container_id = str(
                    current_status.get("containerID", "")
                ).partition("://")[2]
                recovery_worker_restart_count = restarts
                recovery_worker_runtime = verified_pod_container_identity(
                    current,
                    "cpu-ocr-worker",
                    expected_source_tree_sha256=expected_source_fingerprints["worker"],
                    expected_source_profile="worker",
                )
                break
            time.sleep(1)
        write_json(
            args.out / "worker-restart-observations.json",
            {
                "worker_pod": worker_pod_name,
                "worker_pod_uid_before": old_pod_uid,
                "worker_container_id_before": old_container_id,
                "worker_restart_count_before": restart_count_before,
                "recovery_worker_pod": recovery_worker_pod_name,
                "recovery_worker_pod_uid": recovery_worker_pod_uid,
                "recovery_worker_container_id": recovery_worker_container_id,
                "recovery_worker_restart_count": recovery_worker_restart_count,
                "recovery_worker_runtime": recovery_worker_runtime,
                "api_runtime": api_runtime,
                "coordinator_runtime": coordinator_runtime,
                "restarted": restarted,
                "samples": restart_observations,
            },
        )

        recovery_page_log_text = ""
        recovery_log_deadline = min(time.monotonic() + 45, deadline)
        while restarted and time.monotonic() < recovery_log_deadline:
            recovery_log = capture_recovery_log(
                kubeconfig,
                args.namespace,
                recovery_worker_pod_name,
            )
            recovery_page_log_text = str(recovery_log.get("stdout", ""))
            if page_task_execution_success_logged(
                recovery_page_log_text,
                task_id=str(boundary["active_page"].get("celery_task_id") or ""),
                page_num=int(boundary["active_page"]["page_num"]),
            ):
                break
            time.sleep(0.5)
        if recovery_log.get("ok") is True:
            write_bytes_no_replace(
                args.out / "worker-after-restart.log",
                str(recovery_page_log_text).encode("utf-8"),
            )

        failure_stage = "capture_and_validate_final_state"
        while time.monotonic() < deadline:
            final_state = django_state(
                kubeconfig, args.namespace, coordinator_deployment, job_id
            )
            if final_state["job"]["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(1)
        else:
            final_state = django_state(
                kubeconfig, args.namespace, coordinator_deployment, job_id
            )

        api_final = api_status(args.api_url, api_key, args.tenant_id, job_id)
        committed_artifacts: dict[str, Any] = {}
        if final_state["job"]["status"] == "completed":
            committed_artifacts = capture_committed_artifacts(
                kubeconfig,
                expected_context,
                args.namespace,
                args.release,
                job_id,
                submitted["source_sha256"],
                args.out,
            )
        write_json(args.out / "coordinator-final-state.json", final_state)
        write_json(args.out / "coordinator-final-status.json", api_final)
        log_result = capture_logs(
            kubeconfig,
            args.namespace,
            recovery_worker_pod_name or worker_pod_name,
            args.out,
        )
        operator_logs = run(
            [
                *kubectl_base(kubeconfig),
                "logs",
                "-n",
                "keda",
                "deployment/keda-operator",
                "--since=20m",
            ],
            timeout=50,
        )
        if operator_logs["ok"]:
            write_bytes_no_replace(
                args.out / "keda-operator.log",
                operator_logs["stdout"].encode("utf-8"),
            )
        coordinator_logs = run(
            [
                *kubectl_base(kubeconfig),
                "logs",
                "-n",
                args.namespace,
                f"deployment/{coordinator_deployment}",
                "-c",
                "coordinator",
                "--since=20m",
            ],
            timeout=50,
        )
        if coordinator_logs["ok"]:
            write_bytes_no_replace(
                args.out / "coordinator.log",
                coordinator_logs["stdout"].encode("utf-8"),
            )
        rabbit_pod = f"{args.release}-rabbitmq-0"
        rabbit = run(
            [
                *kubectl_base(kubeconfig),
                "exec",
                "-n",
                args.namespace,
                f"pod/{rabbit_pod}",
                "--",
                "rabbitmqctl",
                "list_queues",
                "name",
                "messages",
                "consumers",
            ],
            timeout=30,
        )
        if rabbit["ok"]:
            write_bytes_no_replace(
                args.out / "rabbitmq-queues-final.txt",
                rabbit["stdout"].encode("utf-8"),
            )

        job = final_state["job"]
        pages = final_state["pages"]
        success_pages = [p for p in pages if p["status"] in SUCCESS_PAGE_STATES]
        page_ids: dict[int, list[str]] = {}
        for page in pages:
            page_ids.setdefault(int(page["page_num"]), []).append(
                str(page["celery_task_id"])
            )
        dispatch_keys = [
            (d["stage"], d["generation"]) for d in final_state["dispatches"]
        ]
        active_page_num = int(boundary["active_page"]["page_num"])
        active_task_id = str(boundary["active_page"].get("celery_task_id") or "")
        active_page_row = next(
            (p for p in pages if int(p["page_num"]) == active_page_num), {}
        )
        active_dispatches = [
            d
            for d in final_state["dispatches"]
            if d["stage"] == f"process_page:{active_page_num}"
        ]
        previous_log_text = (
            (args.out / "worker-before-kill.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if (args.out / "worker-before-kill.log").is_file()
            else ""
        )
        current_log_path = args.out / "worker-after-restart.log"
        if not current_log_path.is_file():
            current_log_path = args.out / "worker-current.log"
        current_log_text = (
            current_log_path.read_text(encoding="utf-8", errors="replace")
            if current_log_path.is_file()
            else ""
        )
        recovery_log_capture: dict[str, Any] = {
            "ok": False,
            "path": current_log_path.name,
            "size_bytes": 0,
            "sha256": None,
        }
        if current_log_path.is_file():
            recovery_log_bytes = current_log_path.read_bytes()
            recovery_log_capture = {
                "ok": bool(recovery_log_bytes),
                "path": current_log_path.name,
                "size_bytes": len(recovery_log_bytes),
                "sha256": hashlib.sha256(recovery_log_bytes).hexdigest(),
            }
        log_result["recovery"] = recovery_log_capture
        redelivery = {
            "task_id": active_task_id,
            "same_task_id_in_final_page_row": (
                active_page_row.get("celery_task_id") == active_task_id
            ),
            "same_single_outbox_row": len(active_dispatches) == 1,
            "outbox_broker_task_id_matches": bool(
                active_dispatches
                and active_dispatches[0]["broker_task_id"] == active_task_id
            ),
            "page_completed_on_recovery_worker": bool(
                active_page_row.get("status") in SUCCESS_PAGE_STATES
                and recovery_worker_pod_name
                and active_page_row.get("worker_hostname") == recovery_worker_pod_name
            ),
            "page_execution_logged_after_worker_restart": bool(
                page_task_execution_success_logged(
                    current_log_text,
                    task_id=active_task_id,
                    page_num=active_page_num,
                )
            ),
            "task_received_before_container_loss": bool(
                active_task_id
                and f"Task jobs.tasks.process_page[{active_task_id}] received"
                in previous_log_text
            ),
            "task_received_after_container_restart": bool(
                active_task_id
                and f"Task jobs.tasks.process_page[{active_task_id}] received"
                in current_log_text
            ),
        }
        redelivery["verified"] = all(
            value
            for key, value in redelivery.items()
            if key
            not in {
                "task_id",
                "task_received_before_container_loss",
                "task_received_after_container_restart",
            }
        )
        result = {
            "status": "PASS"
            if (
                boundary.get("kill_command", {}).get("ok") is True
                and restarted
                and recovery_log_capture.get("ok") is True
                and redelivery["verified"]
                and job["status"] == "completed"
                and job["total_pages"]
                == submitted.get("expected_pages", job["total_pages"])
                and job["source_hash"] == submitted["source_sha256"]
                and job["pages_completed"] == job["total_pages"]
                and job["pages_failed"] == 0
                and len(pages) == job["total_pages"]
                and len(success_pages) == job["total_pages"]
                and len(dispatch_keys) == len(set(dispatch_keys))
                and committed_artifacts.get("job_id") == job_id
                and committed_artifacts.get("source_sha256")
                == submitted["source_sha256"]
            )
            else "FAIL",
            "ok": bool(
                boundary.get("kill_command", {}).get("ok") is True
                and restarted
                and recovery_log_capture.get("ok") is True
                and redelivery["verified"]
                and job["status"] == "completed"
                and job["pages_completed"] == job["total_pages"]
                and job["pages_failed"] == 0
                and len(pages) == job["total_pages"]
                and len(success_pages) == job["total_pages"]
                and len(dispatch_keys) == len(set(dispatch_keys))
                and committed_artifacts.get("job_id") == job_id
                and committed_artifacts.get("source_sha256")
                == submitted["source_sha256"]
            ),
            "claim_tier": "single-node-kind-lab",
            "runtime_images": {
                "context": expected_context,
                "namespace": args.namespace,
                "release": args.release,
                "source_fingerprints": expected_source_fingerprints,
                "api": api_runtime,
                "coordinator": coordinator_runtime,
                "worker_before": boundary.get("worker_runtime"),
                "worker_recovery": recovery_worker_runtime,
            },
            "job_id": job_id,
            "source_sha256": submitted["source_sha256"],
            "injected_worker_loss": (
                boundary.get("kill_command", {}).get("ok") is True
            ),
            "worker_restarted": restarted,
            "worker_recovery_pod": recovery_worker_pod_name,
            "worker_recovery_pod_uid": recovery_worker_pod_uid,
            "worker_recovery_container_id": recovery_worker_container_id,
            "worker_recovery_restart_count": recovery_worker_restart_count,
            "worker_log_capture": log_result,
            "job_status": job["status"],
            "total_pages": job["total_pages"],
            "pages_completed": job["pages_completed"],
            "pages_failed": job["pages_failed"],
            "page_result_rows": len(pages),
            "successful_page_rows": len(success_pages),
            "page_task_ids": page_ids,
            "redelivery": redelivery,
            "unique_outbox_stage_generation_pairs": len(set(dispatch_keys)),
            "outbox_rows": len(dispatch_keys),
            "custody_event_rows": len(final_state["custody"]),
            "custody_event_hash_rows": sum(
                bool(e.get("event_hash")) for e in final_state["custody"]
            ),
            "committed_artifacts": committed_artifacts,
            "boundary": {
                "successful_page_count": boundary["successful_page_count"],
                "max_successful_page": boundary["max_successful_page"],
                "active_page": boundary["active_page"],
                "active_task_id": boundary["active_page"].get("celery_task_id"),
                "worker_pod": worker_pod_name,
                "worker_container_id_before": old_container_id,
                "worker_restart_count_before": boundary[
                    "worker_container_restart_count_before"
                ],
                "worker_stop_command_ok": (
                    boundary.get("kill_command", {}).get("ok") is True
                ),
            },
            "limits": [
                "One-node kind lab on Sparks; not multi-host, DGX, real S3/IAM, or production proof.",
                "Runtime Pod image IDs are cross-checked against local Docker image labels for source revision, runtime profile, and source-tree fingerprint.",
                "OCR used the lane recorded in PageResult; this is recovery evidence, not accuracy qualification.",
                "Celery task-received log records are not emitted by this worker image; exact task-id continuity is verified from the page lease, outbox row, recovery container restart, and boundary-page execution log.",
                "Artifact hashes, generation cardinality, and custody-file chain verification require the retained output artifact check.",
            ],
            "artifact_paths": {
                "submit": "coordinator-submit.json",
                "boundary": "worker-kill-boundary.json",
                "restart": "worker-restart-observations.json",
                "final_state": "coordinator-final-state.json",
                "final_status": "coordinator-final-status.json",
                "worker_previous": "worker-previous.log",
                "worker_current": "worker-current.log",
                "worker_recovery": recovery_log_capture.get("path")
                or "worker-after-restart.log",
                "worker_before_kill": "worker-before-kill.log",
                "source_fixture": f"input/{fixture.name}",
                "coordinator_log": "coordinator.log",
                "keda_log": "keda-operator.log",
                "rabbitmq_queues": "rabbitmq-queues-final.txt",
            },
            "started_at": run_started_at,
            "finished_at": now_iso(),
        }
        write_json(args.out / "recovery-result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "job_id": job_id,
                    "worker_restarted": restarted,
                    "job_status": job["status"],
                    "pages_completed": job["pages_completed"],
                    "pages_failed": job["pages_failed"],
                    "boundary_page": boundary["active_page"].get("page_num"),
                }
            )
        )
        return 0 if result["ok"] else 1
    except Exception as exc:
        failure = failure_record(exc, stage=failure_stage)
        write_json(args.out / "recovery-error.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
