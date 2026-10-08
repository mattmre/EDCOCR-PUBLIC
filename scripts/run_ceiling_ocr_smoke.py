#!/usr/bin/env python3
"""Ceiling OCR smoke through the tracked production pipeline entrypoint.

The tracked two-page fixture is processed by the current checkout's
``ocr_gpu_async.py`` inside the model-capable Docker image. Success requires
the canonical searchable PDF, text export, and a valid linked custody chain.
This is a bounded CPU lab E2E; it is not a GPU or production-fleet claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "test-canary-2page.pdf"
DEFAULT_IMAGE = os.environ.get("OCR_CEILING_IMAGE", "ocr-gpu-local:canary")
DEFAULT_TIMEOUT_SECONDS = 900


def _runtime_snapshot(image: str) -> dict:
    """Resolve mutable tags once and bind mounted source separately from image."""
    try:
        from scripts.runtime_source_fingerprint import fingerprint_for_profile
    except ModuleNotFoundError:
        from runtime_source_fingerprint import fingerprint_for_profile
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    rows = json.loads(proc.stdout)
    if len(rows) != 1 or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", rows[0].get("Id", "")
    ):
        raise ValueError("image identity unavailable")
    row = rows[0]
    return {
        "image_id": row["Id"],
        "platform": f"{row.get('Os', '')}/{row.get('Architecture', '')}",
        "labels": row.get("Config", {}).get("Labels") or {},
        "source_sha256": fingerprint_for_profile(REPO_ROOT, "root-cpu"),
    }


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0


def _verify_product_artifacts(output_dir: Path, fixture_name: str) -> dict[str, int]:
    stem = Path(fixture_name).stem
    pdf_path = output_dir / "EXPORT" / "PDF" / fixture_name
    text_path = output_dir / "EXPORT" / "TEXT" / f"{stem}.txt"
    custody_files = list((output_dir / "EXPORT" / "CUSTODY").glob("*.jsonl"))
    if not pdf_path.is_file() or pdf_path.stat().st_size < 1_000:
        raise RuntimeError("canonical production PDF is missing or trivial")
    if not text_path.is_file():
        raise RuntimeError("canonical production text export is missing")
    text = text_path.read_text(encoding="utf-8").strip()
    if len(text) < 20:
        raise RuntimeError("canonical production text export is trivial")
    if len(custody_files) != 1:
        raise RuntimeError("expected exactly one production custody JSONL")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to verify the searchable PDF") from exc
    with fitz.open(pdf_path) as document:
        page_text_lengths = [len(page.get_text().strip()) for page in document]
    if len(page_text_lengths) != 2 or any(length < 10 for length in page_text_lengths):
        raise RuntimeError(
            "production PDF is not a searchable two-page canonical artifact"
        )

    records = []
    for line in custody_files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise RuntimeError("production custody chain is empty")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from ocr_local.features.custody import verify_custody_file

    valid, message = verify_custody_file(str(custody_files[0]))
    if not valid:
        raise RuntimeError(f"production custody verification failed: {message}")
    previous_hash = None
    for index, record in enumerate(records):
        if index == 0:
            if record.get("prev_hash") not in (None, ""):
                raise RuntimeError("production custody genesis is not blank")
        elif record.get("prev_hash") != previous_hash:
            raise RuntimeError("production custody hash linkage is broken")
        previous_hash = record.get("hash")
        if not isinstance(previous_hash, str) or len(previous_hash) != 64:
            raise RuntimeError("production custody event hash is invalid")
    event_types = {record.get("event_type") for record in records}
    ocr_event_types = {"ocr_primary", "ocr_fallback"}
    required = {"file_ingested", "assembly_complete"}
    if not required.issubset(event_types) or not event_types.intersection(
        ocr_event_types
    ):
        raise RuntimeError("production custody chain lacks ingest/OCR/assembly events")
    ocr_pages = {
        record.get("data", {}).get("page_num")
        for record in records
        if record.get("event_type") in ocr_event_types
    }
    if ocr_pages != {1, 2}:
        raise RuntimeError("production custody does not prove OCR for both pages")
    return {
        "pdf_bytes": pdf_path.stat().st_size,
        "text_chars": len(text),
        "custody_events": len(records),
        "searchable_pages": len(page_text_lengths),
    }


def _verify_expected_ocr_route(
    output_dir: Path,
    *,
    expected_engine: str,
    expected_language: str,
) -> dict[str, object]:
    """Require every page custody event to report the requested OCR route."""
    custody_files = list((output_dir / "EXPORT" / "CUSTODY").glob("*.jsonl"))
    if len(custody_files) != 1:
        raise RuntimeError("expected exactly one production custody JSONL")

    ocr_events = []
    for line in custody_files[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event_type") in {"ocr_primary", "ocr_fallback"}:
            ocr_events.append(record.get("data", {}))
    if not ocr_events:
        raise RuntimeError("production custody chain has no OCR events")

    wrong_routes = [
        event
        for event in ocr_events
        if str(event.get("engine", "")).casefold() != expected_engine.casefold()
        or str(event.get("lang_hint", "")).casefold() != expected_language.casefold()
    ]
    if wrong_routes:
        raise RuntimeError(
            f"expected {expected_engine}/{expected_language} for every OCR page; "
            f"observed {wrong_routes}"
        )
    return {
        "engine": expected_engine,
        "language": expected_language,
        "ocr_pages": len(ocr_events),
    }


def run_ceiling(
    *,
    fixture: Path = DEFAULT_FIXTURE,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    evidence_dir: Path | None = None,
    docker_network: str | None = None,
    expected_engine: str | None = None,
    expected_language: str | None = None,
    require_baked_source: bool = False,
) -> int:
    if not fixture.is_file():
        print(f"CEILING_OCR_SMOKE_FAIL missing_fixture {fixture}")
        return 2
    if not _docker_available():
        print("CEILING_OCR_SMOKE_FAIL docker_not_on_path")
        return 2
    if not _image_present(image):
        print(
            f"CEILING_OCR_SMOKE_FAIL image_missing {image} "
            "(build model-capable image or set OCR_CEILING_IMAGE)"
        )
        return 2
    if docker_network not in (None, "none", "bridge"):
        print("CEILING_OCR_SMOKE_FAIL invalid_docker_network")
        return 2
    if bool(expected_engine) != bool(expected_language):
        print("CEILING_OCR_SMOKE_FAIL incomplete_expected_route")
        return 2

    try:
        snapshot = _runtime_snapshot(image)
        image = snapshot["image_id"]
        if require_baked_source:
            labels = snapshot["labels"]
            if (
                labels.get("io.ocr_local.runtime-source-sha256")
                != snapshot["source_sha256"]
                or labels.get("io.ocr_local.runtime-profile") != "root-cpu"
            ):
                raise ValueError("baked source labels do not match current checkout")
    except Exception as exc:
        print(f"CEILING_OCR_SMOKE_FAIL image_source_identity={exc}")
        return 2

    paddle_cache_volume = os.environ.get("OCR_CEILING_PADDLE_VOLUME", "").strip()
    if paddle_cache_volume and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", paddle_cache_volume
    ):
        print("CEILING_OCR_SMOKE_FAIL invalid_paddle_cache_volume")
        return 2

    with tempfile.TemporaryDirectory(prefix="ceiling-product-ocr-") as tmp:
        work = Path(tmp)
        source = work / "source"
        output = work / "output"
        source.mkdir()
        output.mkdir()
        shutil.copy2(fixture, source / fixture.name)
        container_name = f"ocr-ceiling-product-{uuid.uuid4().hex[:12]}"
        mounts = (
            [] if require_baked_source else ["-v", f"{REPO_ROOT}:/workspace-src:ro"]
        )
        runtime_root = "/app" if require_baked_source else "/workspace-src"
        if paddle_cache_volume:
            mounts.extend(["-v", f"{paddle_cache_volume}:/home/ocr/.paddleocr"])
        mounts.extend(["-v", f"{work}:/work"])
        cmd = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            *(["--network", docker_network] if docker_network else []),
            "--entrypoint",
            "python3",
            "-e",
            f"PYTHONPATH={runtime_root}",
            "-e",
            "NUM_WORKERS=1",
            "-e",
            "NUM_EXTRACTORS=1",
            "-e",
            "NUM_COMPRESSORS=1",
            "-e",
            "EXTRACTOR_MODE=thread",
            "-e",
            "SOURCE_FOLDER=/work/source",
            "-e",
            "OUTPUT_FOLDER=/work/output",
            "-e",
            "TEMP_FOLDER=/work/temp",
            "-e",
            "LOG_DIR=/work/logs",
            "-e",
            "FAILURE_REPORT=/work/failures.csv",
            "-e",
            "HEALTHCHECK_FILE=/work/healthcheck",
            *mounts,
            "-w",
            runtime_root,
            image,
            f"{runtime_root}/ocr_gpu_async.py",
            "--source",
            "/work/source",
            "--output",
            "/work/output",
        ]
        print(
            "ceiling_product_entry",
            "ocr_gpu_async.py",
            "fixture",
            fixture.name,
            "image",
            image,
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            sys.stdout.write(exc.stdout or "")
            sys.stderr.write(exc.stderr or "")
            print(
                f"CEILING_OCR_SMOKE_FAIL timeout_after={timeout_seconds}s "
                "container_force_removed"
            )
            return 124
        if proc.returncode != 0:
            sys.stdout.write((proc.stdout or "")[-8_000:])
            sys.stderr.write((proc.stderr or "")[-8_000:])
            print(f"CEILING_OCR_SMOKE_FAIL production_exit={proc.returncode}")
            return proc.returncode or 1
        try:
            evidence = _verify_product_artifacts(output, fixture.name)
            if expected_engine and expected_language:
                evidence["expected_ocr_route"] = _verify_expected_ocr_route(
                    output,
                    expected_engine=expected_engine,
                    expected_language=expected_language,
                )
        except Exception as exc:
            sys.stdout.write((proc.stdout or "")[-8_000:])
            sys.stderr.write((proc.stderr or "")[-8_000:])
            print(f"CEILING_OCR_SMOKE_FAIL artifact_verification={exc}")
            return 1
        if _runtime_snapshot(image)["source_sha256"] != snapshot["source_sha256"]:
            print("CEILING_OCR_SMOKE_FAIL source_changed_during_run")
            return 2
        if evidence_dir is not None:
            target = evidence_dir.expanduser().resolve()
            if target.exists() and any(target.iterdir()):
                print(f"CEILING_OCR_SMOKE_FAIL evidence_dir_not_empty {target}")
                return 2
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output, target, dirs_exist_ok=True)
            input_dir = target / "input"
            input_dir.mkdir(exist_ok=True)
            shutil.copy2(source / fixture.name, input_dir / fixture.name)
            logs_dir = work / "logs"
            if logs_dir.is_dir():
                shutil.copytree(logs_dir, target / "logs", dirs_exist_ok=True)
            if (work / "failures.csv").is_file():
                shutil.copy2(work / "failures.csv", target / "failures.csv")
            (target / "pipeline.stdout.log").write_text(
                proc.stdout or "", encoding="utf-8"
            )
            (target / "pipeline.stderr.log").write_text(
                proc.stderr or "", encoding="utf-8"
            )
            evidence_record = {
                "produced_at_utc": datetime.now(timezone.utc).isoformat(),
                "topology": "single-host CPU lab",
                "runtime_identity": snapshot,
                "source_execution": "baked_image"
                if require_baked_source
                else "read_only_mounted_checkout",
                "docker_network": docker_network or "default",
                "entrypoint": "ocr_gpu_async.py",
                "image": image,
                "fixture": fixture.name,
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "claim_tier": "ceiling_product_e2e_cpu_lab",
                "expected_ocr_route": (
                    {
                        "engine": expected_engine,
                        "language": expected_language,
                    }
                    if expected_engine and expected_language
                    else None
                ),
                "verification": evidence,
            }
            (target / "evidence.json").write_text(
                json.dumps(evidence_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print("ceiling_product_evidence", json.dumps(evidence, sort_keys=True))
        if evidence_dir is not None:
            print("ceiling_product_artifacts", str(evidence_dir.expanduser().resolve()))
        print("SMOKE_TIER=ceiling_product_e2e_cpu_lab")
        print("CEILING_OCR_SMOKE_OK")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--require-baked-source",
        action="store_true",
        help="verify source label and execute baked /app code without source bind mount",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(
            os.environ.get("OCR_CEILING_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="copy verified production outputs and logs into this empty directory",
    )
    parser.add_argument(
        "--docker-network",
        choices=("none", "bridge"),
        help="set the container network namespace; use none for offline proof",
    )
    parser.add_argument(
        "--expected-engine",
        help="require every custody OCR event to name this engine",
    )
    parser.add_argument(
        "--expected-language",
        help="require every custody OCR event to name this language hint",
    )
    args = parser.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be between 30 and 3600")
    return run_ceiling(
        fixture=args.fixture.resolve(),
        image=args.image,
        timeout_seconds=args.timeout_seconds,
        evidence_dir=args.evidence_dir,
        docker_network=args.docker_network,
        expected_engine=args.expected_engine,
        expected_language=args.expected_language,
        require_baked_source=args.require_baked_source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
