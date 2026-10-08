#!/usr/bin/env python3
"""Run deployed-stack E2E validation for the translation-swarm lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    REPO_ROOT / "out/hunter-validation/translation-swarm/20260512-131154/full-long-haul"
)
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "evidence/deployed-stack-e2e"
DEFAULT_MODEL_DIR = REPO_ROOT / "out/translation_models/opus-mt-en-fr-ct2"
DEFAULT_QE_MODEL_DIR = REPO_ROOT / "out/translation_models/cometkiwi-int8"

for _path in (REPO_ROOT, REPO_ROOT / "coordinator"):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout[-8000:],
            "stderr": self.stderr[-8000:],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--qe-model-dir", type=Path, default=DEFAULT_QE_MODEL_DIR)
    parser.add_argument("--api-port", type=int, default=8011)
    parser.add_argument("--frontend-port", type=int, default=3011)
    parser.add_argument("--postgres-port", type=int, default=55432)
    parser.add_argument("--rabbit-port", type=int, default=5672)
    parser.add_argument("--redis-port", type=int, default=56379)
    parser.add_argument("--standalone-rabbit", action="store_true")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument("--keep-services", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def run_cmd(
    name: str,
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> CommandResult:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return CommandResult(name, command, proc.returncode, proc.stdout, proc.stderr)


def resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    if os.name == "nt":
        resolved = shutil.which(f"{name}.cmd")
        if resolved:
            return resolved
    return name


def wait_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        if not port_open("127.0.0.1", port):
            return port
    raise RuntimeError(f"no free TCP port found from {start}")


def wait_http(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def env_for_stack(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    py_path = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "coordinator")])
    postgres_placeholder = "placeholder"
    rabbit_placeholder = "placeholder"
    redis_placeholder = "placeholder"
    django_placeholder = "placeholder"
    flower_placeholder = "placeholder"
    operator_placeholder = "placeholder"
    operator_role = "operator"
    env.update(
        {
            "PYTHONPATH": py_path
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "POSTGRES_PASSWORD": postgres_placeholder,
            "POSTGRES_HOST_PORT": str(args.postgres_port),
            "RABBITMQ_USER": "guest",
            "RABBITMQ_PASSWORD": rabbit_placeholder,
            "REDIS_PASSWORD": redis_placeholder,
            "REDIS_HOST_PORT": str(args.redis_port),
            "FLOWER_PASSWORD": flower_placeholder,
            "DJANGO_SECRET_KEY": django_placeholder,
            "DJANGO_DEBUG": "True",
            "DJANGO_ALLOWED_HOSTS": "localhost,127.0.0.1",
            "DEPLOYMENT_ENV": "development",
            "DATABASE_URL": (
                f"postgres://ocr:{postgres_placeholder}@127.0.0.1:{args.postgres_port}/ocr_coordinator"
            ),
            "CELERY_BROKER_URL": f"amqp://guest:{rabbit_placeholder}@127.0.0.1:{args.rabbit_port}//",
            "CELERY_RESULT_BACKEND": f"redis://:{redis_placeholder}@127.0.0.1:{args.redis_port}/0",
            "REDIS_URL": f"redis://:{redis_placeholder}@127.0.0.1:{args.redis_port}/1",
            "DJANGO_SETTINGS_MODULE": "coordinator.settings",
            "ENABLE_TRANSLATION_API": "true",
            "ALLOW_UNAUTHENTICATED": "true",
            "OCR_API_KEY": operator_placeholder,
            "APIKEY_ROLE": operator_role,
            "CORS_ALLOWED_ORIGINS": f"http://127.0.0.1:{args.frontend_port},http://localhost:{args.frontend_port}",
            "OCR_OUTPUT_DIR": str(args.output_dir / "api-output"),
            "API_DB_PATH": str(
                args.output_dir / "api-output" / f"jobs-{int(time.time())}.db"
            ),
            "TRANSLATION_MODEL_CACHE_DIR": str(
                REPO_ROOT / "out/translation_models/deployed-stack-cache"
            ),
            "TRANSLATION_CT2_MODEL_DIR": str(args.model_dir),
            "NODE_OPTIONS": "--use-system-ca",
        }
    )
    if args.qe_model_dir.exists():
        env.update(
            {
                "OCR_TRANSLATION_QE_ENABLED": "true",
                "OCR_TRANSLATION_QE_MODEL_ID": "solailabs/wmt22-cometkiwi-da-int8",
                "OCR_TRANSLATION_QE_MODEL_PATH": str(args.qe_model_dir),
                "OCR_TRANSLATION_QE_BATCH_SIZE": "1",
                "OCR_TRANSLATION_QE_DEVICE": "cpu",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return env


def sha256_manifest(model_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(model_dir).as_posix()
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def register_ct2_cache(model_dir: Path, env: dict[str, str]) -> CommandResult:
    code = (
        "from pathlib import Path; "
        "from scripts.run_translation_deployed_stack_e2e import sha256_manifest; "
        "from ocr_local.translation.cache import register_model; "
        f"src=Path(r'{model_dir}'); "
        "register_model('local_ct2_opus','local_ct2_opus',src,license='CC-BY-4.0',"
        "nc_licensed=False,sha256_manifest=sha256_manifest(src)); "
        "print('registered local_ct2_opus cache')"
    )
    return run_cmd(
        "register-ct2-cache", [sys.executable, "-c", code], env=env, timeout=180
    )


def start_process(
    name: str,
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._translation_e2e_log_handle = handle  # type: ignore[attr-defined]
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    handle = getattr(proc, "_translation_e2e_log_handle", None)
    if handle is not None:
        handle.close()


def seed_api_records(
    env: dict[str, str], output_dir: Path, progress_log: Path | None = None
) -> dict[str, Any]:
    def progress(message: str) -> None:
        if progress_log is not None:
            append_text(
                progress_log, f"{datetime.now(timezone.utc).isoformat()} seed {message}"
            )

    progress("start")
    os.environ.update(env)
    progress("env-updated")
    from api.database import Job, get_engine, get_session_factory
    from api.review_queue import ReviewQueue
    from ocr_local.features.custody import CustodyChain

    progress("imports-complete")

    job_id = "job_e2e2abc12345"
    job_output = output_dir / "api-output" / job_id
    if job_output.is_dir():
        shutil.rmtree(job_output)
    job_output.mkdir(parents=True, exist_ok=True)
    progress("output-dir-ready")
    chain = CustodyChain(job_id, "e2e2-ocr-input.txt", str(job_output))
    chain.log_event("file_ingested", {"source_path": "fixtures/e2e2-ocr-input.txt"})
    chain.log_event(
        "language_detected",
        {"primary_language": "en", "languages_detected": ["en", "fr"]},
    )
    chain.log_event(
        "TRANSLATION_APPLIED", {"engine_id": "local_ct2_opus", "target_language": "fr"}
    )
    progress("custody-ready")

    get_engine(env["API_DB_PATH"])
    progress("engine-created")
    session = get_session_factory(env["API_DB_PATH"])()
    progress("session-created")
    try:
        existing = session.query(Job).filter(Job.job_id == job_id).first()
        progress("job-query-complete")
        if existing is None:
            existing = Job(
                job_id=job_id,
                status="completed",
                priority="normal",
                source_file="e2e2-ocr-input.txt",
                result_path=str(job_output),
                total_pages=2,
                pages_completed=2,
                current_stage="completed",
            )
            session.add(existing)
        else:
            existing.status = "completed"
            existing.result_path = str(job_output)
            existing.pages_completed = 2
            existing.total_pages = 2
        session.commit()
        progress("job-commit-complete")
    finally:
        session.close()
        progress("session-closed")

    queue = ReviewQueue(env["API_DB_PATH"])
    progress("review-queue-created")
    item = queue.add(
        job_id=job_id,
        reason="low_confidence",
        confidence=0.72,
        quality_classification="review_required",
        metadata={
            "ocr_text": "Hello world. Bonjour tout le monde.",
            "translation": "Bonjour le monde.",
            "events": chain.events,
        },
    )
    browser_item = queue.add(
        job_id=job_id,
        reason="low_confidence",
        confidence=0.74,
        quality_classification="review_required",
        metadata={
            "ocr_text": "Hola mundo. Bonjour tout le monde.",
            "translation": "Bonjour le monde.",
            "events": chain.events,
        },
    )
    progress("review-items-added")
    return {
        "job_id": job_id,
        "review_id": item.review_id,
        "browser_review_id": browser_item.review_id,
        "custody_file": str(job_output / f"{job_id}.custody.jsonl"),
    }


def call_api(
    base_url: str,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    headers = {"X-API-Key": api_key} if api_key else None
    try:
        resp = requests.request(
            method, url, json=json_body, headers=headers, timeout=15
        )
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text[:2000]
        status_code = resp.status_code
    except requests.RequestException as exc:
        body = {"error": type(exc).__name__, "detail": str(exc)}
        status_code = 0
    return {
        "method": method,
        "path": path,
        "status_code": status_code,
        "request": redact_payload(json_body),
        "response": body,
    }


def response_dict(call: dict[str, Any]) -> dict[str, Any]:
    response = call.get("response")
    return response if isinstance(response, dict) else {}


def redact_payload(value: Any) -> Any:
    """Return a JSON-safe copy with credential-like fields redacted."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in {"auth_token", "api_key", "token", "password"}:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def execute_endpoints(
    base_url: str, tenant_id: str, seed: dict[str, Any], *, api_key: str | None = None
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    config_payload = {
        "target_languages": ["fr"],
        "preferred_engines": ["local_ct2_opus"],
        "allow_nc_licensed": False,
        "require_certified": False,
        "default_quality_tier": "standard",
    }
    calls.append(
        call_api(
            base_url,
            "PUT",
            f"/api/v1/translation/tenants/{tenant_id}/config",
            json_body=config_payload,
            api_key=api_key,
        )
    )
    calls.append(
        call_api(
            base_url,
            "GET",
            f"/api/v1/translation/tenants/{tenant_id}/config",
            api_key=api_key,
        )
    )
    glossary_payload = {
        "source_term": f"Party-{int(time.time())}",
        "target_term": "Partie",
        "source_lang": "en",
        "target_lang": "fr",
        "priority": 10,
    }
    create_glossary = call_api(
        base_url,
        "POST",
        f"/api/v1/translation/tenants/{tenant_id}/glossary",
        json_body=glossary_payload,
        api_key=api_key,
    )
    calls.append(create_glossary)
    calls.append(
        call_api(
            base_url,
            "GET",
            f"/api/v1/translation/tenants/{tenant_id}/glossary",
            api_key=api_key,
        )
    )
    entry_id = response_dict(create_glossary).get("id")
    if entry_id:
        calls.append(
            call_api(
                base_url,
                "PATCH",
                f"/api/v1/translation/tenants/{tenant_id}/glossary/{entry_id}",
                json_body={"target_term": "Partie E2E"},
                api_key=api_key,
            )
        )
    calls.append(
        call_api(
            base_url,
            "POST",
            "/api/v1/translation/jobs",
            json_body={
                "source_text": "Hello world. The court file is ready.",
                "source_language": "en",
                "target_languages": ["fr"],
                "tenant_id": tenant_id,
                "use_tenant_router": True,
                "allow_download": False,
            },
            api_key=api_key,
        )
    )
    batch = call_api(
        base_url,
        "POST",
        "/api/v1/translation/batches",
        json_body={
            "tenant_id": tenant_id,
            "source_lang": "en",
            "target_lang": "fr",
            "inputs": [
                {
                    "client_ref": "e2e2-page-1",
                    "text": "Hello world. The court file is ready.",
                }
            ],
        },
        api_key=api_key,
    )
    calls.append(batch)
    batch_id = response_dict(batch).get("batch_id")
    if batch_id:
        for _ in range(20):
            status = call_api(
                base_url,
                "GET",
                f"/api/v1/translation/batches/{batch_id}",
                api_key=api_key,
            )
            calls.append(status)
            if response_dict(status).get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(1)
        calls.append(
            call_api(
                base_url,
                "GET",
                f"/api/v1/translation/batches/{batch_id}/results",
                api_key=api_key,
            )
        )
    calls.append(
        call_api(
            base_url,
            "POST",
            "/api/v1/translation/score-pair",
            json_body={
                "source": "Hello world.",
                "target": "Bonjour le monde.",
                "source_lang": "en",
                "target_lang": "fr",
            },
            api_key=api_key,
        )
    )
    calls.append(
        call_api(
            base_url, "GET", f"/api/v1/jobs/{seed['job_id']}/outputs", api_key=api_key
        )
    )
    calls.append(
        call_api(
            base_url,
            "GET",
            f"/api/v1/jobs/{seed['job_id']}/outputs/custody",
            api_key=api_key,
        )
    )
    calls.append(
        call_api(
            base_url, "GET", f"/api/v1/review/{seed['review_id']}", api_key=api_key
        )
    )
    calls.append(
        call_api(
            base_url,
            "POST",
            f"/api/v1/review/{seed['review_id']}/certify",
            json_body={
                "auth_method": "hardware_token",
                "auth_token": "local-e2e-token",
                "notes": "deployed stack certification attempt",
            },
            api_key=api_key,
        )
    )
    if entry_id:
        calls.append(
            call_api(
                base_url,
                "DELETE",
                f"/api/v1/translation/tenants/{tenant_id}/glossary/{entry_id}",
                api_key=api_key,
            )
        )
    return calls


def endpoint_matrix(calls: list[dict[str, Any]]) -> str:
    rows = [
        ("tenant config", "PUT/GET", "/api/v1/translation/tenants/{tenant_id}/config"),
        (
            "glossary CRUD",
            "POST/GET/PATCH/DELETE",
            "/api/v1/translation/tenants/{tenant_id}/glossary",
        ),
        ("translation submit", "POST", "/api/v1/translation/jobs"),
        ("translation batch status/result", "POST/GET", "/api/v1/translation/batches"),
        ("quality scoring", "POST", "/api/v1/translation/score-pair"),
        ("custody/audit output", "GET", "/api/v1/jobs/{job_id}/outputs/custody"),
        ("review detail", "GET", "/api/v1/review/{review_id}"),
        ("review certify", "POST", "/api/v1/review/{review_id}/certify"),
    ]
    lines = [
        "# Endpoint Coverage Matrix",
        "",
        "This matrix includes deployed-stack E2E execution through the local API, worker, and browser lanes.",
        "",
        "| Area | Method | Endpoint | Executed | Result |",
        "|---|---|---|---|---|",
    ]
    for area, method, endpoint in rows:
        matching = [
            c
            for c in calls
            if c["path"].split("?")[0].startswith(endpoint.split("{")[0])
        ]
        executed = "yes" if matching else "no"
        statuses = (
            ", ".join(str(c["status_code"]) for c in matching) if matching else "n/a"
        )
        result = (
            "passed"
            if matching and all(c["status_code"] < 400 for c in matching)
            else statuses
        )
        lines.append(f"| {area} | {method} | `{endpoint}` | {executed} | {result} |")
    return "\n".join(lines) + "\n"


def write_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    calls: list[dict[str, Any]],
    browser: dict[str, Any] | None,
) -> None:
    write_json(output_dir / "deployed-stack-summary.json", summary)
    write_json(output_dir / "endpoint-requests-responses.json", calls)
    if browser is not None:
        write_json(output_dir / "frontend-browser-result.json", browser)
    blocker_lines = []
    for blocker in summary.get("blockers", []):
        blocker_lines.append(
            f"- **{blocker['name']}**: {blocker['rootCause']} Next action: {blocker['nextAction']}"
        )
    blocker_text = "\n".join(blocker_lines) if blocker_lines else "- None."
    write_text(
        ARTIFACT_ROOT / "deployed-stack-e2e-evidence.md",
        f"""# Deployed-Stack E2E Evidence

Generated: `{datetime.now(timezone.utc).isoformat()}`

## Classification

- Result: **{summary['status']}**
- Product E2E score: **{summary['product_e2e_score']}/100**
- Local remediation BHS score: unchanged at **99/100**
- Production/live BHS score: unchanged at **78/100**

## Stack

- Dependencies: Docker Compose Postgres/RabbitMQ/Redis from `coordinator/docker-compose.coordinator.yml`.
- API: local uvicorn process serving `scripts.translation_e2e_api_app:app`.
- Django models: real coordinator migrations against local Postgres.
- Worker: local Celery worker on `translation_batch` queue when startup succeeds.
- Frontend: local Next dev server when browser lane is enabled.
- CT2: local OPUS-MT bundle registered in the translation cache.
- Quality scoring: local QE model path from `--qe-model-dir` when present; this is local/dev product E2E evidence, not production COMETKiwi approval.
- TSA: not used for E2E-2 endpoint execution; previous local fake/dev TSA evidence remains separate.

## Endpoint Evidence

Request/response payloads: `evidence/deployed-stack-e2e/endpoint-requests-responses.json`

## Blockers

{blocker_text}
""",
    )
    write_text(ARTIFACT_ROOT / "endpoint-coverage-matrix.md", endpoint_matrix(calls))
    browser_status = browser.get("status") if browser else "skipped"
    write_text(
        ARTIFACT_ROOT / "frontend-browser-e2e.md",
        f"""# Frontend Browser E2E

Status: **{browser_status}**

Evidence: `evidence/deployed-stack-e2e/frontend-browser-result.json`

The browser lane starts the frontend against the local API and covers tenant config/glossary, review/certify, and audit/custody pages when the frontend server starts.
""",
    )
    write_text(
        ARTIFACT_ROOT / "stack-run-log.md",
        """# Stack Run Log

Summary JSON: `evidence/deployed-stack-e2e/deployed-stack-summary.json`

Process logs:

- `docker-compose-up.txt`
- `django-migrate.txt`
- `api-uvicorn.log`
- `celery-worker.log`
- `frontend-next.log`
- `browser-e2e.txt`
""",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "stack-run-progress.txt"
    write_text(
        progress_log,
        f"{datetime.now(timezone.utc).isoformat()} start deployed-stack-e2e",
    )
    use_standalone_rabbit = False
    rabbit_container = f"ocr-translation-e2e-rabbit-{os.getpid()}"
    if args.standalone_rabbit:
        if args.rabbit_port == 5672:
            args.rabbit_port = find_free_port(5673)
        use_standalone_rabbit = True
    elif args.rabbit_port == 5672 and port_open("127.0.0.1", 5672):
        args.rabbit_port = find_free_port(5673)
        use_standalone_rabbit = True
    env = env_for_stack(args)
    commands: list[CommandResult] = []
    blockers: list[dict[str, str]] = []
    api_proc = None
    celery_proc = None
    frontend_proc = None
    browser_result = None
    calls: list[dict[str, Any]] = []
    status = "blocked"
    product_e2e_score = 68

    try:
        for name, cmd in (
            ("docker-version", ["docker", "--version"]),
            ("docker-compose-version", ["docker", "compose", "version"]),
        ):
            result = run_cmd(name, cmd, env=env)
            commands.append(result)
            append_text(
                progress_log,
                f"{datetime.now(timezone.utc).isoformat()} {name} rc={result.returncode}",
            )
            if result.returncode != 0:
                blockers.append(
                    {
                        "name": name,
                        "rootCause": result.stderr or result.stdout,
                        "nextAction": "Install/repair Docker and rerun.",
                    }
                )
                raise RuntimeError(name)

        compose_services = (
            ["postgres", "redis"]
            if use_standalone_rabbit
            else ["postgres", "rabbitmq", "redis"]
        )
        if use_standalone_rabbit:
            rabbit_run = run_cmd(
                "docker-run-rabbitmq",
                [
                    "docker",
                    "run",
                    "--rm",
                    "-d",
                    "--name",
                    rabbit_container,
                    "-e",
                    "RABBITMQ_DEFAULT_USER=guest",
                    "-e",
                    f"RABBITMQ_DEFAULT_PASS={env['RABBITMQ_PASSWORD']}",
                    "-p",
                    f"127.0.0.1:{args.rabbit_port}:5672",
                    "rabbitmq:3.13-management-alpine",
                ],
                env=env,
                timeout=120,
            )
            commands.append(rabbit_run)
            append_text(
                progress_log,
                f"{datetime.now(timezone.utc).isoformat()} docker-run-rabbitmq rc={rabbit_run.returncode} port={args.rabbit_port}",
            )
            if rabbit_run.returncode != 0:
                blockers.append(
                    {
                        "name": "dependency-stack",
                        "rootCause": rabbit_run.stderr or rabbit_run.stdout,
                        "nextAction": "Free RabbitMQ port or repair local Docker RabbitMQ image/runtime.",
                    }
                )
                raise RuntimeError("standalone rabbit failed")

        compose = [
            "docker",
            "compose",
            "-f",
            "coordinator/docker-compose.coordinator.yml",
            "up",
            "-d",
            *compose_services,
        ]
        compose_result = run_cmd("docker-compose-up", compose, env=env, timeout=240)
        commands.append(compose_result)
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} docker-compose-up rc={compose_result.returncode} services={','.join(compose_services)}",
        )
        write_text(
            output_dir / "docker-compose-up.txt",
            compose_result.stdout + compose_result.stderr,
        )
        if compose_result.returncode != 0:
            blockers.append(
                {
                    "name": "dependency-stack",
                    "rootCause": compose_result.stderr or compose_result.stdout,
                    "nextAction": "Fix Docker Compose dependency stack startup.",
                }
            )
            raise RuntimeError("compose failed")
        if (
            not wait_port("127.0.0.1", args.postgres_port, 90)
            or not wait_port("127.0.0.1", args.rabbit_port, 90)
            or not wait_port("127.0.0.1", args.redis_port, 90)
        ):
            blockers.append(
                {
                    "name": "dependency-stack-health",
                    "rootCause": "Postgres/RabbitMQ/Redis ports did not become reachable.",
                    "nextAction": "Inspect docker compose ps/logs for unhealthy service.",
                }
            )
            raise RuntimeError("dependency health failed")

        migrate = run_cmd(
            "django-migrate",
            [sys.executable, "coordinator/manage.py", "migrate", "--noinput"],
            env=env,
            timeout=240,
        )
        commands.append(migrate)
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} django-migrate rc={migrate.returncode}",
        )
        write_text(output_dir / "django-migrate.txt", migrate.stdout + migrate.stderr)
        if migrate.returncode != 0:
            blockers.append(
                {
                    "name": "django-migrate",
                    "rootCause": migrate.stderr or migrate.stdout,
                    "nextAction": "Fix coordinator migration/database setup.",
                }
            )
            raise RuntimeError("migrate failed")

        register = register_ct2_cache(args.model_dir, env)
        commands.append(register)
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} register-ct2-cache rc={register.returncode}",
        )
        if register.returncode != 0:
            blockers.append(
                {
                    "name": "ct2-cache-registration",
                    "rootCause": register.stderr or register.stdout,
                    "nextAction": "Ensure local OPUS CT2 bundle exists and has readable files.",
                }
            )
            raise RuntimeError("ct2 registration failed")

        api_proc = start_process(
            "api",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "scripts.translation_e2e_api_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.api_port),
            ],
            output_dir / "api-uvicorn.log",
            cwd=REPO_ROOT,
            env=env,
        )
        if not wait_http(f"http://127.0.0.1:{args.api_port}/api/v1/health", 90):
            blockers.append(
                {
                    "name": "api-start",
                    "rootCause": "Uvicorn did not serve /api/v1/health before timeout.",
                    "nextAction": "Inspect api-uvicorn.log for import/startup failure.",
                }
            )
            raise RuntimeError("api failed")
        append_text(
            progress_log, f"{datetime.now(timezone.utc).isoformat()} api-health ok"
        )

        celery_proc = start_process(
            "celery",
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "coordinator",
                "worker",
                "-Q",
                "translation_batch",
                "-P",
                "solo",
                "-c",
                "1",
                "-l",
                "info",
            ],
            output_dir / "celery-worker.log",
            cwd=REPO_ROOT / "coordinator",
            env=env,
        )
        time.sleep(8)
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} celery-started pid={celery_proc.pid}",
        )

        seed = seed_api_records(env, output_dir, progress_log)
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} seed-api-records job={seed['job_id']} review={seed['review_id']}",
        )
        base_url = f"http://127.0.0.1:{args.api_port}"
        calls = execute_endpoints(
            base_url, "tenant-e2e2", seed, api_key=env["OCR_API_KEY"]
        )
        append_text(
            progress_log,
            f"{datetime.now(timezone.utc).isoformat()} execute-endpoints count={len(calls)}",
        )
        write_json(output_dir / "endpoint-requests-responses.json", calls)

        translation_submit = [
            c for c in calls if c["path"] == "/api/v1/translation/jobs"
        ]
        if translation_submit:
            response = translation_submit[-1].get("response", {})
            message = response.get("message", "") if isinstance(response, dict) else ""
            if "async context" in message:
                blockers.append(
                    {
                        "name": "translation-router-async-context",
                        "rootCause": "Deployed translation job endpoint queued the job but reported router_v2 async-context failure instead of selecting a concrete engine.",
                        "nextAction": "Make tenant-router engine selection async-safe or execute the sync Django lookup through sync_to_async/thread boundary.",
                    }
                )
        batch_submit = [c for c in calls if c["path"] == "/api/v1/translation/batches"]
        if batch_submit and batch_submit[-1]["status_code"] >= 400:
            blockers.append(
                {
                    "name": "translation-batch-submit",
                    "rootCause": f"Deployed batch submit endpoint returned HTTP {batch_submit[-1]['status_code']}: {batch_submit[-1].get('response')}",
                    "nextAction": "Inspect API logs and fix deployed batch creation/worker dispatch path.",
                }
            )
        custody = [c for c in calls if c["path"].endswith("/outputs/custody")]
        if custody and custody[-1]["status_code"] >= 400:
            blockers.append(
                {
                    "name": "custody-output-retrieval",
                    "rootCause": f"Seeded custody chain exists on disk, but deployed custody output endpoint returned HTTP {custody[-1]['status_code']}.",
                    "nextAction": "Persist/register custody artifact metadata so /api/v1/jobs/{job_id}/outputs/custody can retrieve it.",
                }
            )
        review_detail = [
            c
            for c in calls
            if c["path"].startswith("/api/v1/review/")
            and not c["path"].endswith("/certify")
        ]
        if review_detail and review_detail[-1]["status_code"] >= 400:
            blockers.append(
                {
                    "name": "review-detail-auth-or-routing",
                    "rootCause": f"Seeded review detail endpoint returned HTTP {review_detail[-1]['status_code']}: {review_detail[-1].get('response')}",
                    "nextAction": "Provide an operator/admin auth context for E2E or align review route permissions with the browser flow.",
                }
            )
        if any(
            c["path"].endswith("/certify") and c["status_code"] == 404 for c in calls
        ):
            blockers.append(
                {
                    "name": "review-certify-endpoint",
                    "rootCause": "Frontend references /api/v1/review/{id}/certify, but FastAPI review router has no certify route.",
                    "nextAction": "Implement backend certification endpoint or feature-gate/remove the frontend certification action.",
                }
            )
        score_pair = [c for c in calls if c["path"] == "/api/v1/translation/score-pair"]
        if score_pair and score_pair[-1]["status_code"] == 503:
            blockers.append(
                {
                    "name": "quality-scoring-runtime",
                    "rootCause": "Deployed score-pair endpoint executed but COMETKiwi model/runtime is unavailable.",
                    "nextAction": "Provide OCR_TRANSLATION_QE_MODEL_PATH or approved local COMETKiwi runtime artifact.",
                }
            )

        if not args.skip_browser:
            frontend_env = env.copy()
            frontend_env.update(
                {
                    "NEXT_PUBLIC_API_BASE_URL": base_url,
                    "PORT": str(args.frontend_port),
                }
            )
            frontend_proc = start_process(
                "frontend",
                [
                    resolve_executable("npm"),
                    "run",
                    "dev",
                    "--",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(args.frontend_port),
                ],
                output_dir / "frontend-next.log",
                cwd=REPO_ROOT / "frontend",
                env=frontend_env,
            )
            if wait_http(f"http://127.0.0.1:{args.frontend_port}/login", 90):
                browser_env = env.copy()
                browser_env.update(
                    {
                        "TRANSLATION_E2E_FRONTEND_URL": f"http://127.0.0.1:{args.frontend_port}",
                        "TRANSLATION_E2E_BROWSER_OUTPUT": str(
                            output_dir / "frontend-browser-result.json"
                        ),
                        "TRANSLATION_E2E_TENANT_ID": "tenant-e2e2",
                        "TRANSLATION_E2E_REVIEW_ID": seed.get(
                            "browser_review_id", seed["review_id"]
                        ),
                        "TRANSLATION_E2E_JOB_ID": seed["job_id"],
                        "TRANSLATION_E2E_API_KEY": env["OCR_API_KEY"],
                    }
                )
                browser = run_cmd(
                    "browser-e2e",
                    ["node", "scripts/run_translation_deployed_stack_browser_e2e.js"],
                    env=browser_env,
                    timeout=180,
                )
                commands.append(browser)
                write_text(
                    output_dir / "browser-e2e.txt", browser.stdout + browser.stderr
                )
                if (output_dir / "frontend-browser-result.json").exists():
                    browser_result = json.loads(
                        (output_dir / "frontend-browser-result.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for blocker in browser_result.get("blockers", []):
                        blockers.append(blocker)
                    for flow in browser_result.get("flows", []):
                        if flow.get("status") == "failed":
                            blockers.append(
                                {
                                    "name": f"browser-{flow.get('flow', 'unknown')}",
                                    "rootCause": flow.get(
                                        "error", "Browser flow failed."
                                    ),
                                    "nextAction": "Fix the deployed API/auth/data preconditions or frontend flow selectors so browser E2E reaches the expected state.",
                                }
                            )
                else:
                    blockers.append(
                        {
                            "name": "browser-e2e",
                            "rootCause": browser.stderr or browser.stdout,
                            "nextAction": "Inspect browser-e2e.txt and frontend-next.log.",
                        }
                    )
            else:
                blockers.append(
                    {
                        "name": "frontend-start",
                        "rootCause": "Next frontend did not serve /login before timeout.",
                        "nextAction": "Inspect frontend-next.log.",
                    }
                )

        terminal_batch = [
            c
            for c in calls
            if "/api/v1/translation/batches/" in c["path"] and c["path"].count("/") == 5
        ]
        batch_completed = any(
            response_dict(c).get("status") == "completed" for c in terminal_batch
        )
        status = "passed" if not blockers and batch_completed else "partial"
        if status == "passed":
            product_e2e_score = 96
        elif (
            batch_completed
            and blockers
            and all(b.get("name") == "quality-scoring-runtime" for b in blockers)
        ):
            product_e2e_score = 86
        else:
            product_e2e_score = 74
    except Exception as exc:
        if not blockers:
            blockers.append(
                {
                    "name": "deployed-stack-e2e",
                    "rootCause": repr(exc),
                    "nextAction": "Inspect stack-run-log.md and process logs.",
                }
            )
    finally:
        stop_process(frontend_proc)
        stop_process(celery_proc)
        stop_process(api_proc)
        if not args.keep_services:
            stop = run_cmd(
                "docker-compose-stop",
                [
                    "docker",
                    "compose",
                    "-f",
                    "coordinator/docker-compose.coordinator.yml",
                    "stop",
                    "postgres",
                    "rabbitmq",
                    "redis",
                ],
                env=env,
                timeout=120,
            )
            commands.append(stop)
            if use_standalone_rabbit:
                rabbit_stop = run_cmd(
                    "docker-rm-rabbitmq",
                    ["docker", "rm", "-f", rabbit_container],
                    env=env,
                    timeout=60,
                )
                commands.append(rabbit_stop)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "product_e2e_score": product_e2e_score,
        "commands": [c.to_dict() for c in commands],
        "blockers": blockers,
    }
    write_artifacts(output_dir, summary, calls, browser_result)
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
