#!/usr/bin/env python3
"""Exercise durable outbox retry and RabbitMQ restart on an isolated kind lab.

The canary intentionally interrupts only the named ``ocr-closeout-c7`` kind
cluster. It submits an API job while RabbitMQ is stopped, retries the same
idempotency key, leaves the resulting durable task queued with the coordinator
consumer scaled to zero, restarts RabbitMQ's exact container through CRI, then
restores the consumer and verifies one completed artifact generation.

Required environment: ``OCR_COORDINATOR_API_KEY`` (local lab key; never logged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import vtm_pg8_worker_recovery_canary as pg8  # noqa: E402
from runtime_image_identity import verified_pod_container_identity  # noqa: E402
from runtime_source_fingerprint import fingerprint_for_profile  # noqa: E402
from vtm_canary_safety import (  # noqa: E402
    open_no_redirect,
    require_fresh_directory,
    require_isolated_kind_identity,
    require_loopback_http_url,
)

SUCCESS_PAGE_STATES = set(pg8.SUCCESS_PAGE_STATES)
RUNTIME_IDENTITY_FIELDS = (
    "namespace",
    "pod",
    "pod_uid",
    "node",
    "component",
    "container",
    "declared_image",
    "image_id",
    "container_id",
    "ready",
    "docker_image_id",
    "platform",
    "source_revision",
    "source_tree_sha256",
    "source_profile",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def page_worker_pods(state: dict) -> list[str]:
    """Return the distinct worker hosts recorded for successful pages."""
    pages = state.get("pages") if isinstance(state, dict) else None
    if not isinstance(pages, list):
        return []
    return sorted(
        {
            row["worker_hostname"]
            for row in pages
            if isinstance(row, dict)
            and row.get("status") in SUCCESS_PAGE_STATES
            and isinstance(row.get("worker_hostname"), str)
            and row.get("worker_hostname")
        }
    )


def page_workers_match_runtime(state: dict, runtime_images: dict) -> bool:
    worker = runtime_images.get("worker") if isinstance(runtime_images, dict) else None
    return (
        isinstance(worker, dict)
        and page_worker_pods(state) == [worker.get("pod")]
        and bool(worker.get("pod"))
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def kube(kubeconfig: Path) -> list[str]:
    return ["kubectl", "--kubeconfig", str(kubeconfig)]


def command(args: list[str], timeout: int = 30) -> dict:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "argv": args,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": repr(exc), "argv": args}


def kubectl_json(kubeconfig: Path, args: list[str]) -> dict:
    result = command([*kube(kubeconfig), *args, "-o", "json"])
    if not result["ok"]:
        raise RuntimeError(
            f"kubectl {' '.join(args)} failed: {result.get('stderr', '')[-1000:]}"
        )
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kubectl JSON output invalid for {' '.join(args)}") from exc


def ready_component_pod(
    kubeconfig: Path, namespace: str, release: str, component: str, container: str
) -> dict | None:
    """Return one Ready, non-terminating Pod for an exact chart component."""
    pods = kubectl_json(
        kubeconfig,
        [
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app.kubernetes.io/instance={release},"
            f"app.kubernetes.io/component={component}",
        ],
    )
    items = pods.get("items") if isinstance(pods, dict) else None
    matches = []
    for pod in items if isinstance(items, list) else []:
        metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
        status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
        labels = (
            metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        )
        if (
            metadata.get("namespace") != namespace
            or labels.get("app.kubernetes.io/instance") != release
            or labels.get("app.kubernetes.io/component") != component
            or status.get("phase") != "Running"
            or metadata.get("deletionTimestamp")
            or not any(
                row.get("type") == "Ready" and row.get("status") == "True"
                for row in status.get("conditions") or []
                if isinstance(row, dict)
            )
        ):
            continue
        container_statuses = status.get("containerStatuses")
        container_statuses = (
            container_statuses if isinstance(container_statuses, list) else []
        )
        if not any(
            row.get("name") == container and row.get("ready") is True
            for row in container_statuses
            if isinstance(row, dict)
        ):
            continue
        matches.append(pod)
    if len(matches) > 1:
        raise RuntimeError(f"multiple Ready Pods matched component {component!r}")
    return matches[0] if matches else None


def verified_component_identity(
    kubeconfig: Path,
    namespace: str,
    release: str,
    *,
    component: str,
    container: str,
    source_profile: str,
    source_fingerprint: str,
) -> dict | None:
    pod = ready_component_pod(kubeconfig, namespace, release, component, container)
    if pod is None:
        return None
    identity = verified_pod_container_identity(
        pod,
        container,
        expected_source_tree_sha256=source_fingerprint,
        expected_source_profile=source_profile,
    )
    return {key: identity[key] for key in RUNTIME_IDENTITY_FIELDS if key in identity}


def current_source_runtime_images(
    kubeconfig: Path, cluster: str, namespace: str, release: str
) -> dict:
    """Bind the API stack to this checkout before the fault injection."""
    fingerprints = {
        "root-cpu": fingerprint_for_profile(REPO_ROOT, "root-cpu"),
        "coordinator": fingerprint_for_profile(REPO_ROOT, "coordinator"),
        "worker": fingerprint_for_profile(REPO_ROOT, "worker"),
    }
    result = {
        "context": f"kind-{cluster}",
        "namespace": namespace,
        "release": release,
        "source_fingerprints": fingerprints,
        "api": verified_component_identity(
            kubeconfig,
            namespace,
            release,
            component="api",
            container="api",
            source_profile="root-cpu",
            source_fingerprint=fingerprints["root-cpu"],
        ),
        "coordinator": verified_component_identity(
            kubeconfig,
            namespace,
            release,
            component="coordinator",
            container="coordinator",
            source_profile="coordinator",
            source_fingerprint=fingerprints["coordinator"],
        ),
        "worker": None,
    }
    if result["api"] is None or result["coordinator"] is None:
        raise RuntimeError("Ready API and coordinator runtime identities are required")
    return result


def current_worker_runtime_identity(
    kubeconfig: Path, namespace: str, release: str, source_fingerprint: str
) -> dict | None:
    return verified_component_identity(
        kubeconfig,
        namespace,
        release,
        component="cpu-ocr-worker",
        container="cpu-ocr-worker",
        source_profile="worker",
        source_fingerprint=source_fingerprint,
    )


def capture_worker_runtime_identity(
    kubeconfig: Path, namespace: str, release: str, runtime_images: dict
) -> dict | None:
    """Capture the Ready worker identity using this run's worker fingerprint."""
    fingerprints = runtime_images.get("source_fingerprints")
    worker_fingerprint = (
        fingerprints.get("worker") if isinstance(fingerprints, dict) else None
    )
    if not isinstance(worker_fingerprint, str):
        raise RuntimeError("worker source fingerprint is missing")
    identity = current_worker_runtime_identity(
        kubeconfig, namespace, release, worker_fingerprint
    )
    if identity is not None:
        runtime_images["worker"] = identity
    return identity


def scale(kubeconfig: Path, resource: str, replicas: int, namespace: str) -> None:
    result = command(
        [
            *kube(kubeconfig),
            "scale",
            "-n",
            namespace,
            resource,
            f"--replicas={replicas}",
        ]
    )
    if not result["ok"]:
        raise RuntimeError(
            f"could not scale {resource}: {result.get('stderr', '')[-1000:]}"
        )


def wait_workload(
    kubeconfig: Path,
    resource: str,
    replicas: int,
    namespace: str,
    timeout: int = 240,
) -> None:
    if replicas == 0:
        kind, name = resource.split("/", 1)
        field = "status.readyReplicas"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = kubectl_json(kubeconfig, ["get", kind, name, "-n", namespace])
            ready = int((value.get("status") or {}).get("readyReplicas", 0) or 0)
            current = int((value.get("status") or {}).get("replicas", 0) or 0)
            if ready == 0 and current == 0:
                return
            time.sleep(1)
        raise TimeoutError(f"{resource} did not scale to zero ({field})")
    result = command(
        [
            *kube(kubeconfig),
            "rollout",
            "status",
            "-n",
            namespace,
            resource,
            f"--timeout={timeout}s",
        ],
        timeout=timeout + 15,
    )
    if not result["ok"]:
        raise TimeoutError(
            f"{resource} did not become ready: {result.get('stderr', '')[-1000:]}"
        )


def job_by_idempotency(
    kubeconfig: Path, namespace: str, coordinator: str, key_hash: str
) -> dict:
    source = f"""
import os, django, json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from jobs.models import Job, TaskDispatchOutbox
from jobs.rls import control_plane_rls_context
with control_plane_rls_context():
    j = Job.objects.filter(submission_key_hash={key_hash!r}).first()
    if j is None:
        print(json.dumps({{'found': False}}))
    else:
        rows = list(
            TaskDispatchOutbox.objects.filter(job=j)
            .order_by('created_at')
            .values(
                'stage', 'generation', 'status', 'attempts', 'broker_task_id',
                'last_error')
        )
        print(json.dumps({{'found': True, 'job_id': str(j.job_id), 'status': j.status,
          'submission_ready': j.submission_ready,
          'dispatch_generation': j.dispatch_generation,
          'source_hash': j.source_hash, 'total_pages': j.total_pages,
          'pages_completed': j.pages_completed, 'pages_failed': j.pages_failed,
          'outbox': rows}}))
"""
    result = command(
        [
            *kube(kubeconfig),
            "exec",
            "-n",
            namespace,
            f"deployment/{coordinator}",
            "-c",
            "coordinator",
            "--",
            "python3",
            "-c",
            source,
        ]
    )
    if not result["ok"]:
        raise RuntimeError(
            f"idempotency-key database query failed: {result.get('stderr', '')[-1200:]}"
        )
    rows = [
        line for line in result["stdout"].splitlines() if line.strip().startswith("{")
    ]
    if not rows:
        raise RuntimeError("idempotency-key database query produced no JSON")
    return json.loads(rows[-1])


def multipart_submit(
    api_url: str,
    api_key: str,
    tenant_id: str,
    fixture: Path,
    idempotency_key: str,
) -> dict:
    import urllib.error
    import urllib.request

    boundary = f"ocr-broker-outage-{uuid.uuid4().hex}"
    payload = fixture.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{fixture.name}"\r\n'
                "Content-Type: application/pdf\r\n\r\n"
            ).encode(),
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/v1/jobs",
        data=body,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "X-Tenant-ID": tenant_id,
            "Idempotency-Key": idempotency_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with open_no_redirect(req, timeout=45) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        parsed = {"body": raw.decode("utf-8", errors="replace")[:1000]}
    return {
        "http_status": status,
        "payload": parsed,
        "idempotency_key_sha256": hashlib.sha256(idempotency_key.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_size_bytes": len(payload),
        "observed_at": now_iso(),
    }


def rabbit_pod(kubeconfig: Path, namespace: str, release: str) -> dict:
    pod_name = f"{release}-rabbitmq-0"
    pod = kubectl_json(kubeconfig, ["get", "pod", pod_name, "-n", namespace])
    status = pod.get("status") or {}
    metadata = pod.get("metadata") or {}
    node = str((pod.get("spec") or {}).get("nodeName", ""))
    statuses = status.get("containerStatuses") or []
    rabbit_container = next(
        (row for row in statuses if row.get("name") == "rabbitmq"), {}
    )
    pvc_claims = [
        str((volume.get("persistentVolumeClaim") or {}).get("claimName"))
        for volume in ((pod.get("spec") or {}).get("volumes") or [])
        if volume.get("persistentVolumeClaim")
    ]
    return {
        "name": pod_name,
        "uid": metadata.get("uid"),
        "node": node,
        "phase": status.get("phase"),
        "ready": bool(rabbit_container.get("ready")),
        "restart_count": int(rabbit_container.get("restartCount", 0) or 0),
        "container_id": rabbit_container.get("containerID"),
        "pvc_claims": pvc_claims,
    }


def rabbit_queues(kubeconfig: Path, namespace: str, release: str) -> dict:
    result = command(
        [
            *kube(kubeconfig),
            "exec",
            "-n",
            namespace,
            f"pod/{release}-rabbitmq-0",
            "--",
            "rabbitmqctl",
            "--timeout",
            "5",
            "list_queues",
            "name",
            "messages_ready",
            "messages_unacknowledged",
            "consumers",
        ],
        timeout=15,
    )
    if not result["ok"]:
        return {"ok": False, "error": result.get("stderr", "")[-1000:]}
    queues = []
    for line in result["stdout"].splitlines():
        columns = line.split()
        if len(columns) >= 4 and columns[1].isdigit():
            queues.append(
                {
                    "name": columns[0],
                    "messages_ready": int(columns[1]),
                    "messages_unacknowledged": int(columns[2]),
                    "consumers": int(columns[3]),
                }
            )
    return {"ok": True, "queues": queues, "stdout": result["stdout"]}


def wait_rabbit_management(
    kubeconfig: Path,
    namespace: str,
    release: str,
    timeout: int = 180,
) -> dict:
    deadline = time.monotonic() + timeout
    last = {"ok": False, "error": "RabbitMQ management app has not answered yet"}
    while time.monotonic() < deadline:
        last = rabbit_queues(kubeconfig, namespace, release)
        if last.get("ok"):
            return last
        time.sleep(2)
    raise TimeoutError(
        f"RabbitMQ app did not recover for management queries: {last.get('error', '')}"
    )


def stop_rabbit_container(cluster: str, pod: dict) -> dict:
    node = str(pod.get("node", ""))
    runtime, separator, container_id = str(pod.get("container_id", "")).partition("://")
    if (
        runtime != "containerd"
        or not separator
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
    ):
        raise RuntimeError(
            "refusing CRI stop: expected containerd and a full "
            "64-character container ID"
        )
    verified_node = pg8.verify_kind_node(cluster, node)
    result = command(
        [
            "docker",
            "exec",
            verified_node["name"],
            "crictl",
            "stop",
            "--timeout",
            "0",
            container_id,
        ],
        timeout=30,
    )
    return {
        "ok": result.get("ok") is True,
        "method": "stop exact RabbitMQ container through isolated kind-node CRI",
        "node": verified_node["name"],
        "cluster": verified_node["cluster"],
        "container_id": container_id,
        "exit_code": result.get("exit_code"),
        "stderr": result.get("stderr", "")[-800:],
        "stdout": result.get("stdout", "")[-800:],
    }


def restore_original_footprint(
    kubeconfig: Path,
    namespace: str,
    release: str,
    rabbit_replicas: int,
    coordinator_replicas: int,
) -> list[dict[str, str]]:
    """Restore and wait for both workloads, reporting every failed attempt."""
    resources = (
        (
            "rabbitmq",
            "statefulset",
            f"{release}-rabbitmq",
            rabbit_replicas,
        ),
        (
            "celery_coordinator",
            "deployment",
            f"{release}-celery-coordinator",
            coordinator_replicas,
        ),
    )
    errors: list[dict[str, str]] = []
    for label, kind, name, replicas in resources:
        resource = f"{kind}/{name}"
        try:
            current = kubectl_json(kubeconfig, ["get", kind, name, "-n", namespace])
            current_replicas = int((current.get("spec") or {}).get("replicas", 0) or 0)
            if current_replicas != replicas:
                scale(kubeconfig, resource, replicas, namespace)
            # Wait even when spec.replicas already matches: a killed container
            # can still be down while the StatefulSet/Deployment spec is intact.
            wait_workload(kubeconfig, resource, replicas, namespace)
        except Exception as exc:  # noqa: BLE001 — both resources must be attempted
            errors.append({"resource": label, "error": str(exc)})
    return errors


def apply_restoration_outcome(
    evidence: dict[str, Any], errors: list[dict[str, str]], exit_code: int
) -> int:
    evidence["restoration"] = {"ok": not errors, "errors": errors}
    if errors:
        evidence["cleanup_errors"] = errors
        evidence["result"] = "FAIL"
        evidence["ok"] = False
        if exit_code in {0, 1}:
            return 3
    return exit_code


def active_artifact_check(
    kubeconfig: Path, namespace: str, coordinator: str, job_id: str
) -> dict:
    source = f"""
import os, django, json, hashlib, tempfile
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coordinator.settings')
django.setup()
from jobs.models import Job
from jobs.rls import control_plane_rls_context
from jobs.tasks import _get_backend_for_job, _nfs_job_path_for_job
from jobs.artifact_generations import resolve_active_artifacts
with control_plane_rls_context():
    j = Job.objects.get(job_id={job_id!r})
    backend = _get_backend_for_job(j)
    root = _nfs_job_path_for_job(j) if backend.backend_name == 'nfs' else ''
    a = resolve_active_artifacts(job_id=j.job_id, source_file=j.source_file,
      settings_json=j.settings_json, backend=backend.backend_name, job_root=root)
    result = {{
      'backend': a.backend, 'revision': a.revision, 'legacy': a.legacy,
      'pointer': a.pointer,
      'pdf': {{'locator': a.pdf.locator, 'sha256_expected': a.pdf.sha256}},
      'text': {{'locator': a.text.locator, 'sha256_expected': a.text.sha256}}
    }}
    with tempfile.TemporaryDirectory(prefix='ocr-broker-canary-') as tmp:
      for kind, artifact in [('pdf', a.pdf), ('text', a.text)]:
        path = os.path.join(tmp, kind + ('.pdf' if kind == 'pdf' else '.txt'))
        backend.download_file(artifact.locator, path)
        raw = open(path, 'rb').read()
        row = {{'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw),
          'matches_active_pointer': hashlib.sha256(raw).hexdigest() == artifact.sha256}}
        if kind == 'pdf':
          import fitz
          with fitz.open(path) as doc:
            row['page_count'] = doc.page_count
            row['pdf_text_characters'] = sum(
                len(page.get_text().strip()) for page in doc
            )
        else:
          row['text_characters'] = len(raw.decode('utf-8', errors='replace').strip())
        result[kind].update(row)
print(json.dumps(result))
"""
    result = command(
        [
            *kube(kubeconfig),
            "exec",
            "-n",
            namespace,
            f"deployment/{coordinator}",
            "-c",
            "coordinator",
            "--",
            "python3",
            "-c",
            source,
        ],
        timeout=90,
    )
    if not result["ok"]:
        return {"ok": False, "error": result.get("stderr", "")[-1500:]}
    lines = [
        line for line in result["stdout"].splitlines() if line.strip().startswith("{")
    ]
    if not lines:
        return {"ok": False, "error": "artifact verification returned no JSON"}
    value = json.loads(lines[-1])
    value["ok"] = bool(
        value.get("revision")
        and value.get("pdf", {}).get("matches_active_pointer")
        and value.get("pdf", {}).get("page_count")
        and value.get("pdf", {}).get("pdf_text_characters")
        and value.get("text", {}).get("matches_active_pointer")
        and value.get("text", {}).get("text_characters")
    )
    return value


def submit_api(api_url: str, api_key: str) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/v1/health", headers={"X-API-Key": api_key}
    )
    try:
        with open_no_redirect(req, timeout=5) as response:
            body = response.read()
            return {"http_status": response.status, "payload": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {
            "http_status": exc.code,
            "payload": exc.read().decode("utf-8", errors="replace")[:500],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--kubeconfig", required=True, type=Path)
    parser.add_argument("--namespace", default="ocr-local")
    parser.add_argument("--release", default="ocr-kind")
    parser.add_argument("--api-url", default="http://127.0.0.1:18089")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--fault-timeout-sec", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.api_url = require_loopback_http_url(args.api_url, label="--api-url")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    key = os.environ.get("OCR_COORDINATOR_API_KEY", "").strip()
    if not key:
        raise SystemExit("OCR_COORDINATOR_API_KEY is required")
    if not re.fullmatch(r"ocr-closeout-c7(?:-[0-9]{8})?", args.cluster):
        raise SystemExit(
            "refusing cluster: --cluster must name the isolated "
            "ocr-closeout-c7 kind cluster"
        )
    fixture = args.fixture.expanduser().resolve(strict=True)
    kubeconfig = args.kubeconfig.expanduser().resolve(strict=True)
    try:
        output_dir = require_fresh_directory(args.out, root=REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not fixture.is_relative_to(REPO_ROOT):
        raise SystemExit("fixture must be inside this checkout")
    tenant_id = args.tenant_id or f"sparks-broker-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", tenant_id):
        raise SystemExit("tenant id is outside the API's bounded lab format")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = command([*kube(kubeconfig), "config", "current-context"])
    expected_context = f"kind-{args.cluster}"
    if not context["ok"]:
        raise SystemExit(f"refusing kube context: expected {expected_context}")
    nodes = kubectl_json(kubeconfig, ["get", "nodes"])
    node_names = [
        str((row.get("metadata") or {}).get("name", ""))
        for row in nodes.get("items", [])
    ]
    try:
        require_isolated_kind_identity(
            context["stdout"].strip(), args.cluster, node_names
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for node_name in node_names:
        pg8.verify_kind_node(args.cluster, node_name)
    api_health = submit_api(args.api_url, key)
    if api_health.get("http_status") != 200:
        raise SystemExit(f"local coordinator API health failed: {api_health}")
    rabbit_sts = kubectl_json(
        kubeconfig,
        ["get", "statefulset", f"{args.release}-rabbitmq", "-n", args.namespace],
    )
    coordinator_dep = kubectl_json(
        kubeconfig,
        [
            "get",
            "deployment",
            f"{args.release}-celery-coordinator",
            "-n",
            args.namespace,
        ],
    )
    rabbit_original = int((rabbit_sts.get("spec") or {}).get("replicas", 0) or 0)
    coordinator_original = int(
        (coordinator_dep.get("spec") or {}).get("replicas", 0) or 0
    )
    if rabbit_original != 1 or coordinator_original != 1:
        raise SystemExit(
            "expected exactly one local RabbitMQ pod and coordinator "
            "consumer before the canary"
        )
    rabbit_current = rabbit_pod(kubeconfig, args.namespace, args.release)
    if not rabbit_current.get("ready"):
        raise SystemExit("RabbitMQ pod is not ready on the isolated kind node")
    pg8.verify_kind_node(args.cluster, str(rabbit_current.get("node", "")))
    runtime_images = current_source_runtime_images(
        kubeconfig, args.cluster, args.namespace, args.release
    )

    idempotency_key = f"vtm-broker-outage-{uuid.uuid4()}"
    key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "lane": "VTM durable handoff broker interruption",
        "claim_tier": "single-node-kind-lab",
        "started_at": now_iso(),
        "cluster": args.cluster,
        "namespace": args.namespace,
        "release": args.release,
        "tenant_id": tenant_id,
        "fixture": fixture.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "source_size_bytes": fixture.stat().st_size,
        "runtime_images": runtime_images,
        "idempotency_key_sha256": key_hash,
        "api_health": api_health,
        "original_replicas": {
            "rabbitmq": rabbit_original,
            "celery_coordinator": coordinator_original,
        },
        "rabbit_before": rabbit_current,
        "faults": [],
        "limits": [
            "One-node isolated kind lab with local MinIO and a single RabbitMQ broker.",
            "This does not qualify production RabbitMQ HA, real S3/IAM, NFS, "
            "multi-host, GPU, or production OCR accuracy.",
        ],
    }
    exit_code = 2
    try:
        # Stop the only queue consumer and broker; the web/API pod, database,
        # and MinIO remain live so a committed admission can be observed.
        scale(
            kubeconfig,
            f"deployment/{args.release}-celery-coordinator",
            0,
            args.namespace,
        )
        wait_workload(
            kubeconfig,
            f"deployment/{args.release}-celery-coordinator",
            0,
            args.namespace,
        )
        scale(kubeconfig, f"statefulset/{args.release}-rabbitmq", 0, args.namespace)
        wait_workload(
            kubeconfig, f"statefulset/{args.release}-rabbitmq", 0, args.namespace
        )
        evidence["faults"].append(
            {"type": "broker_and_consumer_unavailable", "at": now_iso()}
        )

        first = multipart_submit(args.api_url, key, tenant_id, fixture, idempotency_key)
        evidence["submit_during_outage"] = first
        initial = job_by_idempotency(
            kubeconfig, args.namespace, f"{args.release}-coordinator", key_hash
        )
        if not initial.get("found"):
            raise RuntimeError(
                "API outage attempt did not leave a durable tenant/job record"
            )
        evidence["database_after_outage_submit"] = initial
        initial_dispatch = [
            row for row in initial.get("outbox", []) if row.get("stage") == "api_ingest"
        ]
        if len(initial_dispatch) != 1 or initial_dispatch[0].get("generation") != "0":
            raise RuntimeError(
                "outage admission did not persist exactly one generation-zero "
                "API ingest contract"
            )
        if initial_dispatch[0].get("status") == "published":
            raise RuntimeError(
                "API ingest was published despite the stopped broker; retry "
                "failure was not exercised"
            )

        scale(kubeconfig, f"statefulset/{args.release}-rabbitmq", 1, args.namespace)
        wait_workload(
            kubeconfig, f"statefulset/{args.release}-rabbitmq", 1, args.namespace
        )
        evidence["rabbit_queues_after_statefulset_start"] = wait_rabbit_management(
            kubeconfig, args.namespace, args.release
        )
        evidence["rabbit_restored_at"] = now_iso()

        retry_deadline = time.monotonic() + args.fault_timeout_sec
        replay_attempts = []
        replay = None
        replay_state = None
        while time.monotonic() < retry_deadline:
            replay = multipart_submit(
                args.api_url, key, tenant_id, fixture, idempotency_key
            )
            replay_state = job_by_idempotency(
                kubeconfig, args.namespace, f"{args.release}-coordinator", key_hash
            )
            replay_rows = [
                row
                for row in replay_state.get("outbox", [])
                if row.get("stage") == "api_ingest"
            ]
            replay_attempts.append(
                {
                    "http_status": replay["http_status"],
                    "outbox_status": replay_rows[0].get("status")
                    if len(replay_rows) == 1
                    else None,
                    "observed_at": replay["observed_at"],
                }
            )
            if (
                replay.get("http_status") == 200
                and replay_state.get("found")
                and len(replay_rows) == 1
                and replay_rows[0].get("status") == "published"
            ):
                break
            time.sleep(4)
        evidence["outage_replay_attempts"] = replay_attempts
        evidence["idempotent_replay"] = replay
        if (
            not replay
            or replay.get("http_status") != 200
            or not replay_state
            or not replay_state.get("found")
        ):
            raise RuntimeError(
                "same idempotency key did not recover the admitted API job "
                "after RabbitMQ returned"
            )
        job_id = str(replay.get("payload", {}).get("job_id", ""))
        if not re.fullmatch(r"[0-9a-f-]{36}", job_id):
            raise RuntimeError(
                "idempotent replay did not return a coordinator job UUID"
            )
        evidence["job_id"] = job_id
        state = pg8.django_state(
            kubeconfig, args.namespace, f"{args.release}-coordinator", job_id
        )
        evidence["outbox_after_replay"] = state["dispatches"]
        api_ingest_rows = [
            row for row in state["dispatches"] if row["stage"] == "api_ingest"
        ]
        if (
            len(api_ingest_rows) != 1
            or api_ingest_rows[0]["generation"] != "0"
            or api_ingest_rows[0]["status"] != "published"
        ):
            raise RuntimeError(
                "idempotent retry did not publish the original generation-zero "
                "outbox row"
            )

        # With the consumer held at zero, the coordinator task is a durable
        # ready message. Its availability must survive a Rabbit process crash.
        queues_before = rabbit_queues(kubeconfig, args.namespace, args.release)
        evidence["rabbit_queues_before_restart"] = queues_before
        coordinator_queue = next(
            (
                row
                for row in queues_before.get("queues", [])
                if row["name"] == "coordinator"
            ),
            None,
        )
        if (
            coordinator_queue is None
            or coordinator_queue["messages_ready"] < 1
            or coordinator_queue["consumers"] != 0
        ):
            raise RuntimeError(
                "expected the unconsumed durable ingest message on the "
                "coordinator queue"
            )

        rabbit_before_restart = rabbit_pod(kubeconfig, args.namespace, args.release)
        previous_logs = command(
            [
                *kube(kubeconfig),
                "logs",
                "-n",
                args.namespace,
                f"pod/{args.release}-rabbitmq-0",
                "-c",
                "rabbitmq",
            ],
            timeout=40,
        )
        if previous_logs.get("ok"):
            (output_dir / "rabbitmq-before-restart.log").write_text(
                previous_logs["stdout"], encoding="utf-8"
            )
        restart = stop_rabbit_container(args.cluster, rabbit_before_restart)
        evidence["rabbit_container_restart"] = {
            "before": rabbit_before_restart,
            "injection": restart,
        }
        if not restart["ok"]:
            raise RuntimeError("could not stop the exact isolated RabbitMQ container")
        restart_deadline = time.monotonic() + 180
        rabbit_after_restart = None
        while time.monotonic() < restart_deadline:
            candidate = rabbit_pod(kubeconfig, args.namespace, args.release)
            if (
                candidate.get("ready")
                and int(candidate.get("restart_count", 0))
                > int(rabbit_before_restart.get("restart_count", 0))
                and candidate.get("container_id")
                != rabbit_before_restart.get("container_id")
            ):
                rabbit_after_restart = candidate
                break
            time.sleep(1)
        if rabbit_after_restart is None:
            raise TimeoutError("RabbitMQ did not restart with a new CRI container ID")
        evidence["rabbit_container_restart"]["after"] = rabbit_after_restart
        evidence["rabbit_container_restart"]["restarted"] = True
        queues_after = wait_rabbit_management(kubeconfig, args.namespace, args.release)
        evidence["rabbit_queues_after_restart"] = queues_after
        persisted_queue = next(
            (
                row
                for row in queues_after.get("queues", [])
                if row["name"] == "coordinator"
            ),
            None,
        )
        evidence["queued_message_survived_broker_restart"] = bool(
            persisted_queue
            and persisted_queue["messages_ready"] >= 1
            and persisted_queue["consumers"] == 0
            and rabbit_before_restart.get("pvc_claims")
            == rabbit_after_restart.get("pvc_claims")
        )
        if not evidence["queued_message_survived_broker_restart"]:
            raise RuntimeError(
                "coordinator message or RabbitMQ data volume did not survive "
                "broker restart"
            )

        current_logs = command(
            [
                *kube(kubeconfig),
                "logs",
                "-n",
                args.namespace,
                f"pod/{args.release}-rabbitmq-0",
                "-c",
                "rabbitmq",
            ],
            timeout=40,
        )
        if current_logs.get("ok"):
            (output_dir / "rabbitmq-after-restart.log").write_text(
                current_logs["stdout"], encoding="utf-8"
            )

        scale(
            kubeconfig,
            f"deployment/{args.release}-celery-coordinator",
            coordinator_original,
            args.namespace,
        )
        wait_workload(
            kubeconfig,
            f"deployment/{args.release}-celery-coordinator",
            coordinator_original,
            args.namespace,
        )
        evidence["consumer_restored_at"] = now_iso()

        deadline = time.monotonic() + args.timeout_sec
        final_state = None
        while time.monotonic() < deadline:
            final_state = pg8.django_state(
                kubeconfig, args.namespace, f"{args.release}-coordinator", job_id
            )
            capture_worker_runtime_identity(
                kubeconfig, args.namespace, args.release, runtime_images
            )
            if final_state["job"]["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(3)
        if final_state is None:
            raise RuntimeError("no final coordinator DB state was observed")
        final_status = pg8.api_status(args.api_url, key, tenant_id, job_id)
        evidence["final_state"] = final_state
        evidence["final_api_status"] = final_status
        evidence["page_worker_pods"] = page_worker_pods(final_state)
        evidence["assertions"] = {
            "pages_source_bound_to_ready_worker": page_workers_match_runtime(
                final_state, runtime_images
            )
        }
        if final_state["job"].get("status") == "completed" and not runtime_images.get(
            "worker"
        ):
            raise RuntimeError(
                "completed broker job lacks a Ready source-bound worker identity"
            )
        if (
            final_state["job"]["status"] != "completed"
            or final_status.get("http_status") != 200
        ):
            raise RuntimeError(
                "job did not complete through the restored broker/consumer: "
                f"{final_state['job']['status']}"
            )

        replay_after_completion = multipart_submit(
            args.api_url, key, tenant_id, fixture, idempotency_key
        )
        evidence["idempotent_replay_after_completion"] = replay_after_completion
        replay_state = pg8.django_state(
            kubeconfig, args.namespace, f"{args.release}-coordinator", job_id
        )
        evidence["state_after_completed_replay"] = replay_state
        if (
            replay_after_completion.get("http_status") != 200
            or replay_after_completion.get("payload", {}).get("job_id") != job_id
        ):
            raise RuntimeError("completed job replay did not resolve to the same job")

        pages = final_state.get("pages", [])
        outbox = final_state.get("dispatches", [])
        custody = final_state.get("custody", [])
        stage_pairs = [(row["stage"], row["generation"]) for row in outbox]
        stage_counts: dict[str, int] = {}
        for row in outbox:
            stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1
        assembly_events = [
            row for row in custody if row["event_type"] == "assembly_complete"
        ]
        artifact_evidence = active_artifact_check(
            kubeconfig, args.namespace, f"{args.release}-coordinator", job_id
        )
        evidence["active_artifacts"] = artifact_evidence
        evidence["assertions"].update(
            {
                "initial_broker_outage_reached_durable_outbox": initial_dispatch[0][
                    "status"
                ]
                in {"failed", "pending", "claimed"},
                "idempotent_retry_reused_single_api_ingest_contract": len(
                    api_ingest_rows
                )
                == 1,
                "message_persisted_across_rabbit_container_restart": evidence[
                    "queued_message_survived_broker_restart"
                ],
                "consumer_recovery_completed_job": final_state["job"]["status"]
                == "completed",
                "all_pages_have_success_rows": len(pages)
                == final_state["job"]["total_pages"]
                and all(row["status"] in SUCCESS_PAGE_STATES for row in pages),
                "outbox_stage_generation_pairs_unique": len(stage_pairs)
                == len(set(stage_pairs)),
                "outbox_postprocessing_redrive_published": any(
                    row["stage"] == "postprocessing" and row["status"] == "published"
                    for row in outbox
                ),
                "one_assembly_completion_event": len(assembly_events) == 1,
                "completed_replay_preserved_generation": replay_state["job"][
                    "dispatch_generation"
                ]
                == final_state["job"]["dispatch_generation"]
                and replay_state["job"]["settings_json"].get(
                    "_active_output_generation"
                )
                == final_state["job"]["settings_json"].get("_active_output_generation"),
                "active_pdf_and_text_hashes_verified_from_local_minio": (
                    artifact_evidence.get("ok") is True
                ),
            }
        )
        evidence["outbox_stage_counts"] = stage_counts
        evidence["result"] = "PASS" if all(evidence["assertions"].values()) else "FAIL"
        evidence["ok"] = evidence["result"] == "PASS"
        exit_code = 0 if evidence["ok"] else 1
    except Exception as exc:
        evidence["result"] = "FAIL"
        evidence["ok"] = False
        evidence["error"] = str(exc)
        exit_code = 2
    finally:
        cleanup_errors = restore_original_footprint(
            kubeconfig,
            args.namespace,
            args.release,
            rabbit_original,
            coordinator_original,
        )
        exit_code = apply_restoration_outcome(evidence, cleanup_errors, exit_code)

    evidence["finished_at"] = now_iso()
    evidence_path = output_dir / "broker-outage-live.json"
    write_json(evidence_path, evidence)
    print(
        json.dumps(
            {
                "status": evidence.get("result", "FAIL"),
                "job_id": evidence.get("job_id"),
                "cleanup_ok": evidence.get("restoration", {}).get("ok"),
                "error": evidence.get("error") or evidence.get("cleanup_errors"),
                "evidence": str(evidence_path),
            }
        ),
        file=sys.stdout if exit_code == 0 else sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
