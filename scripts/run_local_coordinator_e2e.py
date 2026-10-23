#!/usr/bin/env python3
"""Replay baked coordinator/CPU images against disposable local PG/broker/MinIO.

This is single-host mock-S3 evidence, never live S3/IAM or multihost proof.
All services belong to a unique Compose project and are removed on exit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import httpx

try:
    from scripts.ocr_rescue_harness_snapshot import (
        capture_rescue_harness,
        load_snapshot_module,
    )
except ModuleNotFoundError:
    from ocr_rescue_harness_snapshot import capture_rescue_harness, load_snapshot_module

try:
    from scripts.runtime_source_fingerprint import fingerprint_for_profile
except ModuleNotFoundError:
    from runtime_source_fingerprint import fingerprint_for_profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _wait_role_health(probe, roles, *, timeout=60, interval=1):
    """Observe Docker's configured healthcheck, retaining failure at deadline."""
    deadline = time.monotonic() + timeout
    observed = probe(roles)
    while not all(status == "healthy" for status in observed.values()) and time.monotonic() < deadline:
        time.sleep(interval)
        observed = probe(roles)
    return observed


def _fanout_cleanup_snapshot(run, job_id: str) -> dict:
    """Read committed PG gauges and passively probe source-cache lease stripes."""
    job_uuid = str(uuid.UUID(job_id))
    code = f"""
import json
from django.db import connection, transaction
from django.db.models import Q, Sum
from jobs.models import Job, PageResult, PageOCRStage, PageOCRByteBudget, PageOCRRasterRetirement
from jobs.rls import control_plane_rls_context
with control_plane_rls_context(), transaction.atomic():
    with connection.cursor() as cursor:
        cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor.execute("SET LOCAL statement_timeout = '5s'")
        cursor.execute("SET LOCAL lock_timeout = '3s'")
    job = Job.objects.get(job_id='{job_uuid}')
    all_stages = PageOCRStage.objects.all()
    all_retirements = PageOCRRasterRetirement.objects.all()
    budget = list(PageOCRByteBudget.objects.filter(pk=1).values_list('reserved_bytes', flat=True))
    stages = list(PageOCRStage.objects.filter(job=job, generation=job.dispatch_generation).order_by('page_num'))
    pages = list(PageResult.objects.filter(job=job).order_by('page_num'))
    snapshot = {{
        'job_id': str(job.job_id), 'job_status': job.status, 'source_sha256': job.source_hash,
        'tenant_id': job.tenant_id, 'generation': job.dispatch_generation,
        'budget_row_count': len(budget), 'global_reserved_bytes': budget[0] if budget else None,
        'all_stage_reserved_bytes': all_stages.aggregate(v=Sum('reserved_bytes'))['v'] or 0,
        'all_retirement_reserved_bytes': all_retirements.aggregate(v=Sum('reserved_bytes'))['v'] or 0,
        'raster_retirement_total': all_retirements.count(),
        'unresolved_raster_retirements': all_retirements.filter(
            Q(reserved_bytes__gt=0) | Q(last_swept_at__isnull=True) | ~Q(last_error='')).count(),
        'global_unsettled_stage_count': all_stages.filter(
            ~Q(status__in=['done','obsolete']) | Q(reserved_bytes__gt=0) |
            ~Q(lease_token='') | Q(lease_expires_at__isnull=False) |
            Q(publish_safe_after__isnull=False)).count(),
        'global_processing_page_count': PageResult.objects.filter(status='processing').count(),
        'processing_page_count': sum(page.status == 'processing' for page in pages),
        'winners': [{{'page': page.page_num, 'status': page.status,
            'generation': page.admitted_generation,
            'document_id': page.document_id,
            'temp_pdf_path': page.temp_pdf_path}} for page in pages],
        'stages': [{{'page': stage.page_num, 'generation': stage.generation,
            'tenant_id': stage.tenant_id, 'source_sha256': stage.source_sha256,
            'status': stage.status, 'reserved_bytes': stage.reserved_bytes,
            'lease_token': stage.lease_token,
            'lease_expires_at': stage.lease_expires_at.isoformat() if stage.lease_expires_at else None,
            'publish_safe_after': stage.publish_safe_after.isoformat() if stage.publish_safe_after else None}}
            for stage in stages],
    }}
print(json.dumps(snapshot, sort_keys=True))
"""
    raw = run("exec", "-T", "api", "python", "manage.py", "shell", "-c", code, timeout=15)
    lines = [line for line in raw.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError("fanout cleanup query did not return exactly one job")
    snapshot = json.loads(lines[0])
    lock_code = r"""
import fcntl, json, os, re, stat
from pathlib import Path
cache = Path(os.environ['S3_CACHE_DIR']) / 'immutable_sources'
root = cache / 'locks'
result = {'observed_stripes': 0, 'busy_stripes': [], 'reservation_files': 0,
          'busy_reservations': [], 'temp_files': [], 'errors': []}
if not root.is_dir() or not cache.is_dir():
    result['errors'].append('cache lease lock directory absent')
else:
    for path in sorted(root.iterdir()):
        if not re.fullmatch(r'[0-9a-f]{2}\.lock', path.name):
            continue
        result['observed_stripes'] += 1
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('unsafe stripe inode')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result['busy_stripes'].append(path.name)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except (OSError, ValueError) as exc:
            result['errors'].append(path.name + ':' + type(exc).__name__)
    for path in sorted(cache.iterdir()):
        if path.name.startswith('.source-') and path.name.endswith(('.tmp', '.json.tmp')):
            result['temp_files'].append(path.name)
        if not path.name.startswith('.reserve-'):
            continue
        result['reservation_files'] += 1
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('unsafe reservation inode')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result['busy_reservations'].append(path.name)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except (OSError, ValueError) as exc:
            result['errors'].append(path.name + ':' + type(exc).__name__)
print(json.dumps(result, sort_keys=True))
"""
    lock_raw = run("exec", "-T", "worker", "python", "-c", lock_code, timeout=15)
    lock_lines = [line for line in lock_raw.splitlines() if line.startswith("{")]
    if len(lock_lines) != 1:
        raise RuntimeError("source-cache lock observer returned no structured result")
    snapshot["source_cache_locks"] = json.loads(lock_lines[0])
    return snapshot


def _fanout_owner_snapshot(run, pids: list[int]) -> dict:
    """Bind infer owners to container identity and kernel process start ticks."""
    if not pids or any(not isinstance(pid, int) or pid <= 0 for pid in pids):
        raise ValueError("infer owner PIDs must be positive integers")
    container_id = run("ps", "-q", "worker", timeout=15).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise RuntimeError("infer worker has no immutable running container ID")
    code = f"""
import json
from pathlib import Path
rows = []
for pid in {sorted(set(pids))!r}:
    stat = (Path('/proc') / str(pid) / 'stat').read_text()
    tail = stat.rsplit(')', 1)[1].split()
    rows.append({{'pid': pid, 'start_ticks': int(tail[19]),
                 'comm': (Path('/proc') / str(pid) / 'comm').read_text().strip()}})
print(json.dumps({{'owners': rows}}))
"""
    raw = run("exec", "-T", "worker", "python", "-c", code, timeout=15)
    lines = [line for line in raw.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError("infer owner observer returned no structured result")
    return {"container_id": container_id, **json.loads(lines[0])}


def _validate_baked_image(profile: str, image: str, info: dict, snapshot: dict | None) -> dict:
    labels = info["Config"].get("Labels") or {}
    current_hash = fingerprint_for_profile(ROOT, profile)
    baked_hash = labels.get("io.ocr_local.runtime-source-sha256")
    if labels.get("io.ocr_local.runtime-profile") != profile or not baked_hash:
        raise RuntimeError(f"{profile} image lacks matching runtime profile/source labels")
    if snapshot is None:
        if baked_hash != current_hash:
            raise RuntimeError(f"{profile} baked source mismatch")
        identity_mode = "current_source"
    else:
        profile_images = snapshot.get("images")
        if isinstance(profile_images, dict):
            expected = profile_images.get(profile)
            if not isinstance(expected, dict) or expected.get("tag") != image:
                raise RuntimeError(f"{profile} image absent from immutable image snapshot")
            if info["Id"] != expected.get("id") or info["Architecture"] != expected.get("architecture"):
                raise RuntimeError(f"{profile} image ID or architecture differs from snapshot")
            if baked_hash != expected.get("runtime_source_sha256"):
                raise RuntimeError(f"{profile} runtime source label differs from snapshot")
        else:
            expected = snapshot.get(image)
            if not isinstance(expected, dict):
                raise RuntimeError(f"{profile} image absent from immutable image snapshot")
            if info["Id"] != expected.get("image_id") or labels != expected.get("labels"):
                raise RuntimeError(f"{profile} image ID or labels differ from snapshot")
            if baked_hash != expected["labels"].get("io.ocr_local.runtime-source-sha256"):
                raise RuntimeError(f"{profile} runtime source label differs from snapshot")
        identity_mode = "pinned_snapshot"
    return {
        "id": info["Id"], "labels": labels, "architecture": info["Architecture"],
        "identity_mode": identity_mode, "current_source_matches": baked_hash == current_hash,
        "current_source_sha256": current_hash,
    }


def _split_presigned_worker_services(app: dict, worker: dict, coordinator_image_id: str) -> tuple[dict, dict]:
    """Keep SDK-capable control tasks separate from credential-free OCR tasks."""
    worker_env = dict(app["environment"])
    for name in (
        "S3_ACCESS_KEY", "S3_SECRET_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ):
        worker_env.pop(name, None)
    worker_env["S3_PRESIGNED_ONLY_WORKER"] = "true"
    worker_env["AWS_EC2_METADATA_DISABLED"] = "true"
    worker_env["S3_CACHE_DIR"] = "/shared/worker-source-cache"
    worker_env["OCR_SOURCE_CACHE_DIAGNOSTICS"] = "true"
    worker_env["OCR_ENGINE_DIAGNOSTICS_PATH"] = "/shared/ocr-engine-events.jsonl"
    worker_env["OCR_ENGINE_DIAGNOSTICS_ROLE"] = "ocr-infer"
    controller = {
        **app,
        "image": coordinator_image_id,
        "command": ["celery", "-A", "coordinator", "worker", "-Q", "coordinator", "-c", "1", "-l", "info"],
        "mem_limit": "4g",
    }
    ocr_worker = {
        **worker,
        "environment": worker_env,
        "command": ["celery", "-A", "coordinator", "worker", "-Q", "ocr_cpu,cpu_general,nlp_general,ocr_tesseract", "-c", "2", "-l", "info"],
    }
    return controller, ocr_worker


def _stage_cpu_worker_service(credential_free_worker: dict) -> dict:
    """Use the OCR image and grant-only environment on CPU stage queues."""
    environment = dict(credential_free_worker["environment"])
    if environment.get("S3_PRESIGNED_ONLY_WORKER") != "true":
        raise ValueError("staged CPU worker must be credential-free")
    environment["OCR_ENGINE_DIAGNOSTICS_ROLE"] = "stage-cpu"
    environment["OCR_WORKER_CAPABILITY"] = "tesseract"
    return {
        **credential_free_worker,
        "environment": environment,
        "command": [
            "celery", "-A", "coordinator", "worker", "-Q",
            "ocr_render,ocr_finalize", "-c", "2", "-l", "info",
        ],
    }


def _stage_tesseract_worker_service(credential_free_worker: dict) -> dict:
    """Keep the cross-engine retry on its own credential-free CPU queue."""
    environment = dict(credential_free_worker["environment"])
    if environment.get("S3_PRESIGNED_ONLY_WORKER") != "true":
        raise ValueError("staged Tesseract worker must be credential-free")
    environment["OCR_ENGINE_DIAGNOSTICS_ROLE"] = "stage-tesseract"
    environment["OCR_WORKER_CAPABILITY"] = "tesseract"
    return {
        **credential_free_worker,
        "environment": environment,
        "command": [
            "celery", "-A", "coordinator", "worker", "-Q",
            "ocr_tesseract", "-c", "1", "-l", "info",
        ],
    }


def _start_container_resource_sampler(project: str, roles: tuple[str, ...], output: Path):
    """Sample Docker cgroup usage for this isolated lab without reading container env."""
    stop_event = threading.Event()
    samples: list[dict] = []
    names = [f"{project}-{role}-1" for role in roles]

    def sample() -> None:
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}", *names],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        row = json.loads(line)
                        samples.append({
                            "at_utc": time.time(), "container": row.get("Name"),
                            "memory_usage": row.get("MemUsage"), "cpu_percent": row.get("CPUPerc"),
                        })
                else:
                    samples.append({"at_utc": time.time(), "sample_error": "docker_stats_unavailable"})
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                samples.append({"at_utc": time.time(), "sample_error": "docker_stats_unavailable"})
            stop_event.wait(3)

    thread = threading.Thread(target=sample, name="ocr-lab-resource-sampler", daemon=True)
    thread.start()

    def finish() -> None:
        stop_event.set()
        thread.join(timeout=25)
        output.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        scale = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1048576,
                 "kB": 1000 / 1048576, "MB": 1000000 / 1048576, "GB": 1000000000 / 1048576}
        peaks: dict[str, dict[str, float]] = {}
        for row in samples:
            role = row.get("container")
            if not role:
                continue
            peak = peaks.setdefault(role, {"memory_usage_mib": 0.0, "cpu_percent": 0.0})
            used = str(row.get("memory_usage") or "").split("/", 1)[0].strip()
            for unit in sorted(scale, key=len, reverse=True):
                if used.endswith(unit):
                    try:
                        peak["memory_usage_mib"] = max(peak["memory_usage_mib"], float(used[:-len(unit)]) * scale[unit])
                    except ValueError:
                        pass
                    break
            try:
                peak["cpu_percent"] = max(peak["cpu_percent"], float(str(row.get("cpu_percent")).rstrip("%")))
            except (TypeError, ValueError):
                pass
        output.with_name("container-resource-peaks.json").write_text(
            json.dumps({"scope": "observed_docker_cgroup_samples_not_rss_guarantee",
                        "interval_seconds": 3, "samples": len(samples), "peaks": peaks}, indent=2) + "\n",
            encoding="utf-8",
        )

    return finish


def _outbox_beat_service(app: dict, coordinator_image_id: str) -> dict:
    """Exercise the configured durable outbox relay during staged lab runs."""
    return {
        **app,
        "image": coordinator_image_id,
        "command": [
            "celery", "-A", "coordinator", "beat", "-l", "info",
            "--schedule", "/tmp/celerybeat-schedule",
        ],
        "mem_limit": "1g",
    }


def _configure_celery_role_healthchecks(services: dict) -> None:
    """Probe the actual Celery role instead of inheriting an API image probe."""
    for role in ("worker", "coordinator-worker", "stage-cpu-worker", "stage-tesseract-worker"):
        service = services.get(role)
        if service is None:
            continue
        command = service["command"]
        if command[:4] != ["celery", "-A", "coordinator", "worker"]:
            raise ValueError(f"{role} is not a Celery worker")
        if "-n" not in command and "--hostname" not in command:
            command.extend(["-n", f"{role}@%h"])
        service["healthcheck"] = {
            "test": ["CMD", "python3", "/app/coordinator/worker_healthcheck.py"],
            "interval": "15s", "timeout": "15s", "start_period": "60s", "retries": 3,
        }
    beat = services.get("outbox-beat")
    if beat is not None:
        if beat["command"][:4] != ["celery", "-A", "coordinator", "beat"]:
            raise ValueError("outbox-beat is not Celery Beat")
        # Beat has no worker nodename to ping. Check its PID 1 role and that
        # its durable schedule has been written recently enough to keep firing.
        script = (
            "import glob,pathlib,time,sys; "
            "argv=pathlib.Path('/proc/1/cmdline').read_bytes().decode().split('\\0'); "
            "files=glob.glob('/tmp/celerybeat-schedule*'); "
            "sys.exit(0 if 'celery' in argv and 'beat' in argv "
            "and any(time.time()-pathlib.Path(p).stat().st_mtime<600 for p in files) else 1)"
        )
        beat["healthcheck"] = {
            "test": ["CMD", "python3", "-c", script],
            "interval": "30s", "timeout": "10s", "start_period": "90s", "retries": 3,
        }


def wait_for_stable_readiness(
    probe, *, timeout=100, stable_successes=1, retry_interval=2, sleep=time.sleep
):
    """Require spaced consecutive successful probes, resetting on any failure."""
    deadline = time.monotonic() + timeout
    successes = 0
    last_output = None
    while time.monotonic() < deadline:
        try:
            last_output = probe()
            successes += 1
            if successes >= stable_successes:
                return last_output
        except Exception:
            successes = 0
        sleep(retry_interval)
    raise TimeoutError(
        f"readiness probe did not pass {stable_successes} consecutive times; "
        f"last output={last_output!r}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-image", required=True)
    parser.add_argument("--coordinator-image", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retain-workdir", action="store_true", help="Keep the isolated Compose and source/output artifacts.")
    parser.add_argument("--corpus-manifest", type=Path, help="Replay the fixed rescue corpus in this isolated coordinator lab.")
    parser.add_argument("--corpus-only", action="store_true", help="Run the corpus without the separate broker-outage canary.")
    parser.add_argument("--corpus-timeout-per-job", type=float, default=600)
    parser.add_argument("--fanout-pages", type=int, default=0, help="Run a separate real raster-page fanout probe; use 100 for P01/P04.")
    parser.add_argument("--fanout-rounds", type=int, default=3, help="Cold/warm worker-process pairs of the fanout probe.")
    parser.add_argument("--presigned-workers", action="store_true", help="Use job-bound presigned URLs for worker object I/O in the mock MinIO lab.")
    parser.add_argument("--staged-page-ocr", action="store_true", help="Route render and finalize to separate credential-free CPU consumers in the presigned mock MinIO lab.")
    parser.add_argument("--stage-worker-loss-probe", action="store_true", help="Kill a live inference worker and prove automatic staged recovery through Beat.")
    parser.add_argument("--native-worker-loss-probe", action="store_true", help="Kill a real skip-OCR worker after one committed native page and prove broker redelivery.")
    parser.add_argument("--native-only", action="store_true", help="Run native worker-loss and text-input probes without replaying the already qualified OCR corpus.")
    parser.add_argument("--fair-admission-only", action="store_true", help="Run the isolated Postgres/RabbitMQ 1/2/16/32 scheduling rehearsal without OCR consumers.")
    parser.add_argument("--image-snapshot", type=Path, help="Pin both baked image IDs and labels to a reviewed immutable snapshot while checkout moves.")
    parser.add_argument(
        "--storage-faults",
        action="store_true",
        help="Also replay isolated restore and retention failure boundaries",
    )
    args = parser.parse_args()
    if args.corpus_only and args.corpus_manifest is None:
        parser.error("--corpus-only requires --corpus-manifest")
    if args.corpus_timeout_per_job <= 0:
        parser.error("--corpus-timeout-per-job must be positive")
    if args.fanout_pages and (args.corpus_manifest is None or not args.presigned_workers):
        parser.error("--fanout-pages requires --corpus-manifest and --presigned-workers")
    if args.staged_page_ocr and not args.presigned_workers:
        parser.error("--staged-page-ocr requires --presigned-workers")
    if args.stage_worker_loss_probe and (not args.staged_page_ocr or args.corpus_manifest is None):
        parser.error("--stage-worker-loss-probe requires --staged-page-ocr and --corpus-manifest")
    if args.native_worker_loss_probe and (not args.presigned_workers or args.corpus_manifest is None):
        parser.error("--native-worker-loss-probe requires --presigned-workers and --corpus-manifest")
    if args.native_only and not args.native_worker_loss_probe:
        parser.error("--native-only requires --native-worker-loss-probe")
    if args.fanout_pages and (args.fanout_pages < 21 or args.fanout_rounds < 3):
        parser.error("fanout proof requires at least 21 pages and three rounds")
    if args.fair_admission_only and (args.corpus_manifest or args.stage_worker_loss_probe or args.fanout_pages):
        parser.error("--fair-admission-only cannot be combined with corpus or page probes")
    args.out.mkdir(parents=True, exist_ok=False)
    frozen_fair_probe = None
    fair_probe_sha256 = None
    if args.fair_admission_only:
        source = ROOT / "scripts/probe_ocr_fair_admission_live.py"
        frozen_path = args.out / "fair-admission-probe-snapshot.py"
        shutil.copy2(source, frozen_path)
        fair_probe_sha256 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        frozen_fair_probe = load_snapshot_module(frozen_path, "_ocr_frozen_fair_admission_probe")
    frozen_runner = None
    harness_snapshot = None
    if args.corpus_manifest is not None:
        args.corpus_manifest, harness_snapshot, frozen_validator = capture_rescue_harness(
            args.out,
            args.corpus_manifest.resolve(),
            scripts={
                "run_local_coordinator_e2e.py": Path(__file__),
                "run_ocr_rescue_coordinator_corpus.py": ROOT / "scripts/run_ocr_rescue_coordinator_corpus.py",
                "validate_ocr_rescue_outputs.py": ROOT / "scripts/validate_ocr_rescue_outputs.py",
                "runtime_source_fingerprint.py": ROOT / "scripts/runtime_source_fingerprint.py",
                "vtm_minio_retention_canary.py": ROOT / "scripts/vtm_minio_retention_canary.py",
                "vtm_minio_backup_restore_canary.py": ROOT / "scripts/vtm_minio_backup_restore_canary.py",
                "custody.py": ROOT / "ocr_local/features/custody.py",
            },
        )
        frozen_runner = load_snapshot_module(
            args.out / "harness-snapshot/scripts/run_ocr_rescue_coordinator_corpus.py",
            "_ocr_rescue_frozen_coordinator_runner",
        )
        frozen_runner.ROOT = ROOT
        for name in ("validate", "_ratio", "_validate_manifest", "normalize"):
            setattr(frozen_runner, name, getattr(frozen_validator, name))
    snapshot = None
    if args.image_snapshot is not None:
        snapshot = json.loads(args.image_snapshot.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("image snapshot must be an object")
    result = {
        "ok": False,
        "tier": (
            "single_host_mock_pg_rabbit_fair_scheduling_no_ocr"
            if args.fair_admission_only else "single_host_mock_s3_coordinator_real_ocr"
        ),
        "images": {},
    }
    if harness_snapshot is not None:
        result["harness_snapshot"] = harness_snapshot
        result["frozen_manifest_path"] = str(args.corpus_manifest)
    if fair_probe_sha256 is not None:
        result["fair_admission_probe_sha256"] = fair_probe_sha256
    if args.image_snapshot is not None:
        result["image_snapshot_sha256"] = hashlib.sha256(args.image_snapshot.read_bytes()).hexdigest()
    for profile, image in [
        ("root-cpu", args.worker_image),
        ("coordinator", args.coordinator_image),
    ]:
        info = json.loads(
            subprocess.check_output(["docker", "image", "inspect", image])
        )[0]
        result["images"][profile] = _validate_baked_image(profile, image, info, snapshot)
    result["harness_sha256"] = (
        harness_snapshot["scripts_sha256"]["run_local_coordinator_e2e.py"]
        if harness_snapshot is not None
        else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    result["collector_sha256"] = {
        name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
        for name in [
            "vtm_minio_retention_canary.py",
            "vtm_minio_backup_restore_canary.py",
        ]
    }
    project = "ocr-closeout-" + uuid.uuid4().hex[:10]
    key, password, admin_password, s3_password = (secrets.token_hex(24) for _ in range(4))
    sensitive = [key, password, admin_password, s3_password]
    workdir = (
        nullcontext(tempfile.mkdtemp(prefix=project))
        if args.retain_workdir
        else tempfile.TemporaryDirectory(prefix=project)
    )
    with workdir as tmp:
        directory = Path(tmp)
        if args.retain_workdir:
            result["retained_workdir"] = str(directory)
        shared = directory / "shared"
        shared.mkdir(mode=0o777)
        shared.chmod(0o777)
        (shared / "worker-source-cache").mkdir(mode=0o777)
        (shared / "worker-source-cache").chmod(0o777)
        directory.chmod(0o755)
        env = {
            "DJANGO_SECRET_KEY": key,
            "CELERY_TASK_SIGNING_KEY": key,
            "PII_ENCRYPTION_KEY": base64.urlsafe_b64encode(
                secrets.token_bytes(32)
            ).decode(),
            "WEBHOOK_SECRET_KEY": key,
            "OCR_COORDINATOR_API_KEY": key,
            "DJANGO_ALLOWED_HOSTS": "localhost,127.0.0.1",
            "DEPLOYMENT_ENV": "development",
            "DATABASE_URL": f"postgres://ocr_app:{password}@postgres:5432/ocr",
            "CELERY_BROKER_URL": f"amqp://ocr:{password}@rabbitmq:5672//",
            "CELERY_RESULT_BACKEND": "redis://redis:6379/0",
            "REDIS_URL": "redis://redis:6379/1",
            "NFS_ROOT": "/shared",
            "STORAGE_BACKEND": "s3",
            "S3_ENDPOINT": "http://minio:9000",
            "S3_BUCKET": "ocr-closeout",
            "S3_ACCESS_KEY": "ocrcloseout",
            "S3_SECRET_KEY": s3_password,
            "S3_REGION": "us-east-1",
            "OCR_TASK_ROUTING": "cpu",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": "/app/coordinator:/app",
        }
        if args.presigned_workers:
            env.update({
                "S3_USE_PRESIGNED_URLS": "true",
                "S3_PRESIGNED_TASK_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
                "S3_PRESIGNED_ALLOW_PRIVATE": "true",
                "S3_PRESIGNED_ALLOW_HTTP": "true",
            })
            sensitive.append(env["S3_PRESIGNED_TASK_KEY"])
        if args.staged_page_ocr:
            env["OCR_STAGED_PAGE_OCR_ENABLED"] = "true"
        if args.stage_worker_loss_probe:
            env["OCR_STAGE_LEASE_SECONDS"] = "60"
            env["OCR_STAGE_RECOVERY_INTERVAL_SECONDS"] = "5"
        if args.native_worker_loss_probe:
            env["OCR_NATIVE_PROCESSING_LEASE_SECONDS"] = "20"
        sensitive.append(env["PII_ENCRYPTION_KEY"])
        app = {
            "environment": env,
            "volumes": [f"{shared}:/shared"],
            "working_dir": "/app/coordinator",
        }
        services = {
            "postgres": {
                "image": "postgres:16-alpine",
                "environment": {
                    "POSTGRES_USER": "ocr_admin",
                    "POSTGRES_PASSWORD": admin_password,
                    "POSTGRES_DB": "ocr",
                },
            },
            "rabbitmq": {
                "image": "rabbitmq:3.13-management-alpine",
                "environment": {
                    "RABBITMQ_DEFAULT_USER": "ocr",
                    "RABBITMQ_DEFAULT_PASS": password,
                },
            },
            "redis": {"image": "redis:7-alpine"},
            "minio": {
                "image": "quay.io/minio/minio:RELEASE.2025-06-13T11-33-47Z",
                "command": ["server", "/data"],
                "environment": {
                    "MINIO_ROOT_USER": "ocrcloseout",
                    "MINIO_ROOT_PASSWORD": s3_password,
                },
            },
            "api": {
                **app,
                "image": result["images"]["coordinator"]["id"],
                "ports": ["127.0.0.1::8000"],
                "command": [
                    "gunicorn",
                    "coordinator.wsgi:application",
                    "--bind",
                    "0.0.0.0:8000",
                    "--workers",
                    "1",
                    "--timeout",
                    "120",
                ],
            },
            "worker": {
                **app,
                "environment": env | {
                    "S3_CACHE_DIR": "/shared/worker-source-cache",
                    "OCR_SOURCE_CACHE_DIAGNOSTICS": "true",
                    "OCR_ENGINE_DIAGNOSTICS_PATH": "/shared/ocr-engine-events.jsonl",
                },
                "image": result["images"]["root-cpu"]["id"],
                "command": [
                    "celery",
                    "-A",
                    "coordinator",
                    "worker",
                    "-Q",
                    "coordinator,ocr_cpu,cpu_general,nlp_general,ocr_tesseract",
                    "-c",
                    "2",
                    "-l",
                    "info",
                ],
                "mem_limit": "8g",
            },
        }
        if args.presigned_workers:
            services["coordinator-worker"], services["worker"] = _split_presigned_worker_services(
                app, services["worker"], result["images"]["coordinator"]["id"]
            )
        if args.staged_page_ocr:
            services["worker"]["command"][5] = "ocr_cpu,cpu_general,nlp_general"
            # A cold/warm pair has 200 page inferences. Keep both prefork
            # children alive for that bounded pair, then recycle finitely.
            services["worker"]["command"].append("--max-tasks-per-child=512")
            services["worker"]["environment"]["OCR_WORKER_CAPABILITY"] = "paddle"
            services["worker"]["environment"].update({
                "OCR_NATIVE_THREADS_PER_PREDICTOR": "1",
                "OCR_ENGINE_REPLICAS_PER_KEY": "2",
                "OCR_INFERENCE_SLOTS_PER_DEVICE": "2",
            })
            services["stage-cpu-worker"] = _stage_cpu_worker_service(services["worker"])
            services["stage-tesseract-worker"] = _stage_tesseract_worker_service(services["worker"])
            services["outbox-beat"] = _outbox_beat_service(
                app, result["images"]["coordinator"]["id"]
            )
        if args.storage_faults:
            services["restore-minio"] = dict(services["minio"])
        _configure_celery_role_healthchecks(services)
        control_roles = ("api", "coordinator-worker", "outbox-beat") if args.staged_page_ocr else (
            ("api", "coordinator-worker") if args.presigned_workers else ("api",)
        )
        ocr_roles = ("worker", "stage-cpu-worker", "stage-tesseract-worker") if args.staged_page_ocr else ("worker",)
        for role in control_roles:
            if services[role]["image"] != result["images"]["coordinator"]["id"]:
                raise RuntimeError(f"{role} does not use the pinned coordinator image")
        for role in ocr_roles:
            if services[role]["image"] != result["images"]["root-cpu"]["id"]:
                raise RuntimeError(f"{role} does not use the pinned OCR image")
        result["service_image_roles"] = {
            role: services[role]["image"] for role in (*control_roles, *ocr_roles)
        }
        config = directory / "compose.json"
        config.write_text(json.dumps({"services": services}))
        config.chmod(0o600)
        base = ["docker", "compose", "-p", project, "-f", str(config)]

        def run(*command, check=True, timeout=180):
            proc = subprocess.run(
                base + list(command), capture_output=True, text=True, timeout=timeout
            )
            if check and proc.returncode:
                message = proc.stdout + proc.stderr
                for value in sensitive:
                    message = message.replace(value, "[REDACTED]")
                raise RuntimeError(message[-12000:])
            return proc.stdout

        def wait_ready(command, timeout=100, stable_successes=1):
            return wait_for_stable_readiness(
                command, timeout=timeout, stable_successes=stable_successes
            )

        def shell(code):
            return run("exec", "-T", "api", "python", "manage.py", "shell", "-c", code)

        def role_health(roles):
            observed = {}
            for role in roles:
                container = f"{project}-{role}-1"
                probe = subprocess.run(
                    ["docker", "inspect", "--format", "{{json .State.Health}}", container],
                    capture_output=True, text=True, timeout=15,
                )
                if probe.returncode:
                    raise RuntimeError(f"cannot inspect {role} health")
                health = json.loads(probe.stdout)
                if not isinstance(health, dict):
                    raise RuntimeError(f"{role} has no configured healthcheck")
                observed[role] = health.get("Status")
            return observed

        resource_sampler_stop = None
        try:
            run("up", "-d", "--pull", "never", "postgres", "rabbitmq", "redis", "minio")
            wait_ready(
                lambda: run(
                    "exec",
                    "-T",
                    "postgres",
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "ocr_admin",
                    "-d",
                    "ocr",
                ),
                stable_successes=3,
            )
            run(
                "exec",
                "-T",
                "postgres",
                "psql",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                "ocr_admin",
                "-d",
                "ocr",
                "-c",
                f"CREATE ROLE ocr_app LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS; ALTER SCHEMA public OWNER TO ocr_app; GRANT CONNECT, CREATE ON DATABASE ocr TO ocr_app;",
            )
            wait_ready(
                lambda: run(
                    "exec", "-T", "rabbitmq", "rabbitmq-diagnostics", "-q", "ping"
                )
            )
            run(
                "run",
                "--rm",
                "--no-deps",
                "api",
                "python",
                "manage.py",
                "migrate",
                "--noinput",
            )
            run("up", "-d", "--pull", "never", "api")
            shell(
                "import boto3; from django.conf import settings; boto3.client('s3',endpoint_url=settings.S3_ENDPOINT,aws_access_key_id=settings.S3_ACCESS_KEY,aws_secret_access_key=settings.S3_SECRET_KEY).create_bucket(Bucket=settings.S3_BUCKET)"
            )
            if args.fair_admission_only:
                result["fair_admission"] = frozen_fair_probe.replay_fair_admission_probe(
                    shell, run, args.out / "fair-admission",
                )
                result["ok"] = result["fair_admission"]["passed"]
                return 0 if result["ok"] else 1
            if args.presigned_workers:
                workers = ["coordinator-worker", "worker"]
                if args.staged_page_ocr:
                    workers.extend(("stage-cpu-worker", "stage-tesseract-worker", "outbox-beat"))
                run("up", "-d", "--pull", "never", *workers)
            else:
                run("up", "-d", "--pull", "never", "worker")
            health_roles = ("api", "coordinator-worker", "worker", "stage-cpu-worker", "stage-tesseract-worker", "outbox-beat") if args.staged_page_ocr else (
                ("api", "coordinator-worker", "worker") if args.presigned_workers else ("api", "worker")
            )
            def healthy_roles():
                statuses = role_health(health_roles)
                if any(status != "healthy" for status in statuses.values()):
                    raise RuntimeError("service health still starting")
                return statuses

            result["service_health_initial"] = wait_ready(
                healthy_roles,
                timeout=150, stable_successes=2,
            )
            if args.corpus_manifest is not None:
                resource_sampler_stop = _start_container_resource_sampler(
                    project,
                    ("api", "coordinator-worker", "worker", "stage-cpu-worker", "stage-tesseract-worker", "outbox-beat")
                    if args.staged_page_ocr else ("api", "worker"),
                    args.out / "container-resource-samples.jsonl",
                )
            address = run("port", "api", "8000").strip()
            headers = {
                "X-API-Key": key,
                "X-Tenant-ID": "closeout-lab",
                "Idempotency-Key": "closeout-replay",
            }
            fixture = (ROOT / "tests/fixtures/test-canary-2page.pdf").read_bytes()
            with httpx.Client(
                base_url="http://" + address, headers=headers, timeout=120
            ) as client:

                def health():
                    response = client.get("/api/v1/health")
                    response.raise_for_status()

                wait_ready(health)

                if args.native_only:
                    result["native_worker_loss"] = frozen_runner.replay_native_worker_loss_probe(
                        client, args.out / "native-worker-loss", shared, shell, run,
                    )
                    result["native_text_inputs"] = frozen_runner.replay_native_text_inputs_probe(
                        client, args.out / "native-text-inputs", shared, shell, run,
                    )
                    result["service_health_post_run"] = _wait_role_health(role_health, health_roles)
                    result["ok"] = (
                        result["native_worker_loss"]["passed"]
                        and result["native_text_inputs"]["passed"]
                        and all(status == "healthy" for status in result["service_health_post_run"].values())
                    )
                    return 0 if result["ok"] else 1

                if args.corpus_manifest is not None:
                    def cache_metrics():
                        raw = run(
                            "exec", "-T", "worker", "python", "-c",
                            "import json, os; from pathlib import Path; from jobs.source_cache import ImmutableSourceCache; print(json.dumps(ImmutableSourceCache(str(Path(os.environ['S3_CACHE_DIR']) / 'immutable_sources')).diagnostic_counters()))",
                        )
                        return json.loads([line for line in raw.splitlines() if line.startswith("{")][-1])

                    corpus = frozen_runner.replay_corpus(
                        client,
                        args.corpus_manifest,
                        args.out / "corpus",
                        shared,
                        shell,
                        timeout_per_job=args.corpus_timeout_per_job,
                        cache_metrics=cache_metrics,
                        presigned_worker=args.presigned_workers,
                    )
                    result["corpus"] = {
                        "scope": corpus["scope"],
                        "quality_passed": corpus["quality_passed"],
                        "manifest_sha256": corpus["manifest_sha256"],
                        "presigned_workers": args.presigned_workers,
                        "staged_page_ocr": args.staged_page_ocr,
                        "report": "corpus/quality.json",
                    }
                    if args.stage_worker_loss_probe and corpus["quality_passed"]:
                        result["stage_worker_loss"] = frozen_runner.replay_stage_worker_loss_probe(
                            client, args.corpus_manifest, args.out / "stage-worker-loss",
                            shared, shell, run,
                        )
                    if args.native_worker_loss_probe and corpus["quality_passed"]:
                        result["native_worker_loss"] = frozen_runner.replay_native_worker_loss_probe(
                            client, args.out / "native-worker-loss", shared, shell, run,
                        )
                        result["native_text_inputs"] = frozen_runner.replay_native_text_inputs_probe(
                            client, args.out / "native-text-inputs", shared, shell, run,
                        )
                    if args.fanout_pages and corpus["quality_passed"]:
                        def engine_events():
                            path = shared / "ocr-engine-events.jsonl"
                            if not path.exists():
                                return []
                            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

                        result["fanout"] = frozen_runner.replay_fanout_probe(
                            client,
                            args.corpus_manifest,
                            args.out / "fanout",
                            cache_metrics=cache_metrics,
                            engine_events=engine_events,
                            reset_worker=lambda: run("restart", "worker"),
                            shared=shared,
                            shell=shell,
                            cleanup_snapshot=lambda job_id: _fanout_cleanup_snapshot(run, job_id),
                            owner_snapshot=lambda pids: _fanout_owner_snapshot(run, pids),
                            pages=args.fanout_pages,
                            rounds=args.fanout_rounds,
                            timeout_per_job=max(args.corpus_timeout_per_job, 1200),
                        )
                    if args.corpus_only:
                        result["service_health_post_corpus"] = role_health(health_roles)
                        result["ok"] = (
                            corpus["quality_passed"]
                            and (not args.fanout_pages or result.get("fanout", {}).get("passed", False))
                            and (not args.stage_worker_loss_probe or result.get("stage_worker_loss", {}).get("passed", False))
                            and (not args.native_worker_loss_probe or result.get("native_worker_loss", {}).get("passed", False))
                            and (not args.native_worker_loss_probe or result.get("native_text_inputs", {}).get("passed", False))
                            and all(
                                status == "healthy"
                                for status in result["service_health_post_corpus"].values()
                            )
                        )
                        return 0 if result["ok"] else 1

                def submit(content=fixture):
                    return client.post(
                        "/api/v1/jobs",
                        files={"file": ("canary.pdf", content, "application/pdf")},
                    )

                # A committed source and generation-zero outbox must survive
                # loss of the broker. Only this disposable project's broker stops.
                run("stop", "rabbitmq")
                response = submit()
                response.raise_for_status()
                job_id = response.json()["job_id"]
                result["job_id"] = job_id
                admission = run(
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-At",
                    "-U",
                    "ocr_admin",
                    "-d",
                    "ocr",
                    "-c",
                    f"SELECT submission_ready FROM jobs_job WHERE job_id='{job_id}'; SELECT generation || ':' || status FROM jobs_taskdispatchoutbox WHERE job_id='{job_id}' AND stage='api_ingest';",
                )
                (args.out / "broker-outage-admission.txt").write_text(admission)
                assert "t" in admission.splitlines(), admission
                assert any(
                    line in ("0:failed", "0:pending") for line in admission.splitlines()
                ), admission
                run("start", "rabbitmq")
                wait_ready(
                    lambda: run(
                        "exec", "-T", "rabbitmq", "rabbitmq-diagnostics", "-q", "ping"
                    )
                )
                time.sleep(10)
                relay = shell(
                    "from jobs.tasks import relay_task_dispatch_outbox; print(relay_task_dispatch_outbox.delay().get(timeout=60))"
                )
                (args.out / "broker-recovery-relay.txt").write_text(relay)
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    state_response = client.get(f"/api/v1/jobs/{job_id}")
                    state_response.raise_for_status()
                    state = state_response.json()
                    if state["status"] in ("completed", "failed", "cancelled"):
                        break
                    time.sleep(3)
                (args.out / "job.json").write_text(json.dumps(state, indent=2))
                assert state["status"] == "completed", state
                assert state["total_pages"] == state["pages_completed"] == 2, state
                replay = submit()
                assert replay.status_code in (200, 201, 202), replay.text
                assert replay.json()["job_id"] == job_id
                assert submit(fixture + b"\nchanged").status_code == 409
                assert (
                    client.get(
                        f"/api/v1/jobs/{job_id}", headers={"X-Tenant-ID": "other-lab"}
                    ).status_code
                    == 404
                )
                bundle_response = client.get(f"/api/v1/jobs/{job_id}/document-bundle")
                bundle_response.raise_for_status()
                bundle = bundle_response.json()
                (args.out / "document-bundle.json").write_text(
                    json.dumps(bundle, indent=2)
                )
                assert (
                    bundle["source_file_sha256"] == hashlib.sha256(fixture).hexdigest()
                )
                assert sum(len(span["text"]) for span in bundle["spans"]) > 100
                assert (
                    isinstance(bundle["custody_chain_head"], str)
                    and len(bundle["custody_chain_head"]) == 64
                )
                result["bundle_text_sha256"] = bundle["artifact_manifest"]["artifacts"][
                    0
                ]["sha256"]
                assert [span["page_number"] for span in bundle["spans"]] == [1, 2]
                assert all(span["metadata"]["location_scope"] == "page_text_artifact" for span in bundle["spans"])
                assert (
                    hashlib.sha256("\n\n".join(span["text"] for span in bundle["spans"]).encode()).hexdigest()
                    == result["bundle_text_sha256"]
                )
                result["checks"] = [
                    "broker_outage_durable_admission_and_recovery",
                    "real_ocr_two_pages",
                    "idempotent_submission",
                    "changed_source_conflict",
                    "tenant_isolation",
                    "document_bundle",
                ]
                artifact_verification = shell(f"""
import json, tempfile, hashlib, fitz
from pathlib import Path
from jobs.models import Job, CustodyEvent
from jobs.rls import worker_tenant_rls_context
from jobs.tasks import _get_backend_for_job, _build_custody_records
from jobs.artifact_generations import resolve_active_artifacts
with worker_tenant_rls_context('closeout-lab'):
    job = Job.objects.get(job_id={job_id!r})
    backend = _get_backend_for_job(job)
    artifacts = resolve_active_artifacts(job_id=job.job_id, source_file=job.source_file, settings_json=job.settings_json, backend='s3', job_root='')
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / 'output.pdf'
        backend.download_file(artifacts.pdf.locator, str(target), max_bytes=20_000_000)
        data = target.read_bytes()
        with fitz.open(target) as pdf:
            assert len(pdf) == 2 and all(len(page.get_text().strip()) > 20 for page in pdf)
        events = list(CustodyEvent.objects.filter(job=job).order_by('id'))
        assert events and all(event.event_hash for event in events)
        records = _build_custody_records(events)
        assert records[-1]['hash'] == {bundle["custody_chain_head"]!r}
        print(json.dumps({{'ok': True, 'searchable_pdf_pages': 2, 'pdf_sha256': hashlib.sha256(data).hexdigest(), 'custody_records': records}}))
""")
                verified_artifacts = json.loads(
                    [
                        line
                        for line in artifact_verification.splitlines()
                        if line.startswith("{")
                    ][-1]
                )
                (args.out / "artifact-verification.json").write_text(
                    json.dumps(verified_artifacts, indent=2)
                )
                result["checks"].extend(
                    ["searchable_pdf_download", "sealed_custody_hash_reverification"]
                )
                if args.storage_faults:
                    run("up", "-d", "--pull", "never", "restore-minio")
                    run(
                        "cp",
                        str(ROOT / "scripts/vtm_minio_backup_restore_canary.py"),
                        "api:/tmp/backup_helpers.py",
                    )
                    backup = shell("""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, '/tmp')
import backup_helpers as helper
from django.conf import settings
source = helper._client(settings.S3_ENDPOINT, settings.S3_ACCESS_KEY, settings.S3_SECRET_KEY, settings.S3_REGION)
target = helper._client('http://restore-minio:9000', settings.S3_ACCESS_KEY, settings.S3_SECRET_KEY, settings.S3_REGION)
with tempfile.TemporaryDirectory() as tmp:
    archive = Path(tmp) / 'backup.tar'
    manifest, total, count, source_bytes, skipped = helper._export(source, settings.S3_BUCKET, archive, 100_000_000, 100, 20_000_000)
    verified = helper._restore(target, settings.S3_BUCKET, archive, manifest)
    assert verified == count and skipped == 0
    print(json.dumps({'ok': True, 'verified_objects': verified, 'bytes': total, 'manifest': manifest, 'scope': 'two disposable MinIO containers on one host; object restore only'}))
""")
                    backup_value = json.loads(
                        [line for line in backup.splitlines() if line.startswith("{")][
                            -1
                        ]
                    )
                    (args.out / "object-restore.json").write_text(
                        json.dumps(backup_value, indent=2)
                    )
                    # Reuse the collector's failure-boundary code on this runner's
                    # own disposable namespace. Its kind-only CLI stays unchanged.
                    import sys

                    sys.path.insert(0, str(ROOT / "scripts"))
                    from vtm_minio_retention_canary import (
                        retention_fault_matrix_source,
                        safe_fault_matrix,
                    )

                    fault_output = shell(
                        retention_fault_matrix_source("closeout-lab", job_id)
                    )
                    fault_value = json.loads(
                        [
                            line
                            for line in fault_output.splitlines()
                            if line.startswith("{")
                        ][-1]
                    )
                    safe_value = safe_fault_matrix(fault_value)
                    (args.out / "retention-fault-matrix.json").write_text(
                        json.dumps(safe_value, indent=2)
                    )
                    assert safe_value["ok"], safe_value
                    result["checks"].extend(
                        [
                            "separate_minio_container_object_restore",
                            "retention_four_failure_boundaries_and_retry",
                        ]
                    )
                result["service_health_post_run"] = role_health(health_roles)
                result["ok"] = (
                    (not args.corpus_manifest or result["corpus"]["quality_passed"])
                    and all(status == "healthy" for status in result["service_health_post_run"].values())
                )
        except Exception as exc:
            message = str(exc)
            for value in sensitive:
                message = message.replace(value, "[REDACTED]")
            result["error"] = message
            raise RuntimeError(message) from None
        finally:
            try:
                if resource_sampler_stop is not None:
                    resource_sampler_stop()
                if "service_health_initial" in result:
                    try:
                        result["service_health_final"] = role_health(health_roles)
                    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                        result["service_health_final_error"] = type(exc).__name__
                logs = run("logs", "--no-color", check=False, timeout=30)
                for value in sensitive:
                    logs = logs.replace(value, "[REDACTED]")
                (args.out / "services.log").write_text(logs)
                (args.out / "result.json").write_text(json.dumps(result, indent=2))
            finally:
                # Evidence errors (including full disk) must not leak services.
                run("down", "-v", "--remove-orphans", timeout=120)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
