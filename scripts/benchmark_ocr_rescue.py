#!/usr/bin/env python3
"""Run externally supplied OCR commands against the rescue corpus.

Each variant command receives ``{manifest}``, ``{run_dir}``, ``{phase}``, and
``{repeat}`` substitutions and must write ``outputs.json`` in its run dir.
It should also write ``runtime_identity.json`` with observed source_commit,
image_id, service_instance_id, and model_instance_id. Repeated subprocesses
alone are not evidence that a model stayed warm.
Optional ``metrics.json`` may report actual stage times, init counts, I/O
bytes, utilization, and job latencies. Missing measurements stay unavailable.
Commands are argv arrays, never shell strings. This runner starts no service
on its own and cannot convert a failed quality run into useful throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.validate_ocr_rescue_outputs import validate  # noqa: E402

SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")
MEASURED_FIELDS = (
    "model_init_count",
    "remote_bytes",
    "local_copy_bytes",
    "cpu_percent",
    "gpu_util_percent",
    "rss_peak_bytes",
    "vram_peak_bytes",
    "queue_age_s",
    "render_s",
    "preprocess_s",
    "inference_wait_s",
    "inference_s",
    "overlay_s",
    "persistence_s",
    "finalization_s",
    "compression_s",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(config: dict) -> None:
    if int(config.get("repetitions", 0)) < 3:
        raise ValueError("at least three benchmark repetitions are required")
    if not isinstance(config.get("experimental_variable"), str) or not config["experimental_variable"]:
        raise ValueError("experimental_variable must name the one varied contract field")
    if not isinstance(config.get("competing_workloads"), str) or not config["competing_workloads"]:
        raise ValueError("competing_workloads must be recorded")
    variants = config.get("variants", {})
    if set(variants) != {"baseline", "candidate"}:
        raise ValueError("benchmark requires baseline and candidate variants")
    variable = config["experimental_variable"]
    contracts = []
    for name, variant in variants.items():
        if not COMMIT_ID.fullmatch(str(variant.get("source_commit", ""))):
            raise ValueError(f"{name} source_commit must be a full committed SHA")
        if not SHA256_ID.fullmatch(str(variant.get("image_id", ""))):
            raise ValueError(f"{name} image_id must be an immutable sha256 digest")
        command = variant.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(part, str) for part in command):
            raise ValueError(f"{name} command must be a nonempty argv list")
        contract = variant.get("contract")
        if not isinstance(contract, dict) or variable not in contract:
            raise ValueError(f"{name} contract must include experimental_variable")
        if any(field not in contract for field in ("model_hashes", "resources", "settings")) or not contract["model_hashes"]:
            raise ValueError(f"{name} contract requires model hashes, resources, and settings")
        for model_hash in contract.get("model_hashes", {}).values():
            if not SHA256_ID.fullmatch(str(model_hash)):
                raise ValueError("model asset IDs must be immutable sha256 digests")
        contracts.append({key: value for key, value in contract.items() if key != variable})
    if contracts[0] != contracts[1]:
        raise ValueError("baseline/candidate fixture, model, settings or resource contract differs outside intended variable")
    reset = config.get("cold_reset_command")
    if reset is not None and (not isinstance(reset, list) or not reset or any(not isinstance(part, str) for part in reset)):
        raise ValueError("cold_reset_command must be an argv list")


def _substitute(command: list[str], *, manifest: Path, run_dir: Path, phase: str, repeat: int, variant_name: str, variant: dict) -> list[str]:
    values = {
        "manifest": str(manifest), "run_dir": str(run_dir), "phase": phase,
        "repeat": str(repeat), "variant": variant_name,
        "source_commit": variant["source_commit"], "image_id": variant["image_id"],
    }
    expanded = []
    for part in command:
        for name, value in values.items():
            part = part.replace("{" + name + "}", value)
        expanded.append(part)
    return expanded


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)


def _measured(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {name: None for name in MEASURED_FIELDS} | {"job_latency_p50_s": None, "job_latency_p95_s": None, "metrics_status": "unavailable"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    observed = {name: metrics.get(name) for name in MEASURED_FIELDS}
    for name, value in observed.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"metrics {name} must be a nonnegative measured number")
        if name in {"model_init_count", "remote_bytes", "local_copy_bytes", "rss_peak_bytes", "vram_peak_bytes"} and not isinstance(value, int):
            raise ValueError(f"metrics {name} must be an integer count")
    latencies = metrics.get("job_latencies_s", [])
    if not isinstance(latencies, list) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("metrics job_latencies_s must be nonnegative measured values")
    observed["job_latency_p50_s"] = _percentile(latencies, 0.5)
    observed["job_latency_p95_s"] = _percentile(latencies, 0.95)
    observed["metrics_status"] = "reported"
    observed["metrics_sha256"] = _sha256(metrics_path)
    observed["gpu_device"] = metrics.get("gpu_device")
    return observed


def _identity(path: Path, variant: dict) -> dict[str, Any]:
    if not path.is_file():
        return {"identity_status": "unavailable", "service_instance_id": None, "model_instance_id": None}
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise ValueError("runtime identity must be an object")
    valid = (
        observed.get("source_commit") == variant["source_commit"]
        and observed.get("image_id") == variant["image_id"]
        and all(isinstance(observed.get(key), str) and observed[key] for key in ("service_instance_id", "model_instance_id"))
    )
    return {
        "identity_status": "matched" if valid else "mismatch",
        "service_instance_id": observed.get("service_instance_id"),
        "model_instance_id": observed.get("model_instance_id"),
        "runtime_identity_sha256": _sha256(path),
    }


def run(config_path: Path, manifest_path: Path, output_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("benchmark output root must be empty; preserve prior attempts")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    manifest_hash = _sha256(manifest_path)
    truth = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not truth.get("cases"):
        raise ValueError("benchmark manifest has no cases")
    useful_truth_pages = sum(
        bool(page["expected_text"].strip())
        for case in truth["cases"]
        for page in case["pages"]
    )
    records = []
    for variant_name in ("baseline", "candidate"):
        variant = config["variants"][variant_name]
        for phase in ("cold", "warm"):
            if phase == "cold" and not config.get("cold_reset_command"):
                records.append({"variant": variant_name, "phase": phase, "status": "unavailable_no_cold_reset", "repeat": None})
                continue
            for repetition in range(1, int(config["repetitions"]) + 1):
                run_dir = output_root / variant_name / phase / f"repeat-{repetition}"
                run_dir.mkdir(parents=True, exist_ok=False)
                if phase == "cold":
                    reset_argv = _substitute(config["cold_reset_command"], manifest=manifest_path, run_dir=run_dir, phase=phase, repeat=repetition, variant_name=variant_name, variant=variant)
                    reset = subprocess.run(reset_argv, cwd=config.get("cwd") or None, capture_output=True, text=True, check=False)
                    (run_dir / "reset.stdout.txt").write_text(reset.stdout, encoding="utf-8")
                    (run_dir / "reset.stderr.txt").write_text(reset.stderr, encoding="utf-8")
                    if reset.returncode != 0:
                        records.append({"variant": variant_name, "phase": phase, "repeat": repetition, "status": "cold_reset_failed", "reset_returncode": reset.returncode})
                        continue
                command = _substitute(variant["command"], manifest=manifest_path, run_dir=run_dir, phase=phase, repeat=repetition, variant_name=variant_name, variant=variant)
                start = time.perf_counter()
                completed = subprocess.run(command, cwd=config.get("cwd") or None, capture_output=True, text=True, check=False)
                wall = time.perf_counter() - start
                (run_dir / "command.stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (run_dir / "command.stderr.txt").write_text(completed.stderr, encoding="utf-8")
                record: dict[str, Any] = {
                    "variant": variant_name,
                    "phase": phase,
                    "repeat": repetition,
                    "status": "command_failed" if completed.returncode else "completed",
                    "command_returncode": completed.returncode,
                    "wall_seconds": round(wall, 6),
                    "source_commit": variant["source_commit"],
                    "image_id": variant["image_id"],
                    "declared_contract": variant["contract"],
                    "manifest_sha256": manifest_hash,
                    "run_dir": str(run_dir),
                }
                record.update(_measured(run_dir / "metrics.json"))
                record.update(_identity(run_dir / "runtime_identity.json", variant))
                mapping = run_dir / "outputs.json"
                if completed.returncode == 0 and mapping.is_file():
                    quality = validate(manifest_path, mapping)
                    (run_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    record["quality_passed"] = quality["passed"]
                    if quality["passed"] and wall > 0:
                        useful_pages = useful_truth_pages
                        record["useful_pages"] = useful_pages
                        record["useful_pages_per_second"] = round(useful_pages / wall, 6)
                    else:
                        record["useful_pages"] = 0
                        record["useful_pages_per_second"] = None
                        record["status"] = "quality_failed"
                else:
                    record["quality_passed"] = False
                    record["useful_pages"] = 0
                    record["useful_pages_per_second"] = None
                    if completed.returncode == 0:
                        record["status"] = "outputs_missing"
                records.append(record)
    summary = {}
    for name in ("baseline", "candidate"):
        warm = [item["wall_seconds"] for item in records if item["variant"] == name and item["phase"] == "warm" and item.get("quality_passed")]
        summary[name] = {
            "quality_valid_warm_repetitions": len(warm),
            "wall_median_s": round(statistics.median(warm), 6) if warm else None,
            "wall_p95_s": _percentile(warm, 0.95),
            "wall_range_s": round(max(warm) - min(warm), 6) if warm else None,
        }
        warm_records = [item for item in records if item["variant"] == name and item["phase"] == "warm" and item.get("repeat") is not None]
        summary[name]["warm_model_reuse_proven"] = (
            len(warm_records) >= 3
            and all(item.get("identity_status") == "matched" for item in warm_records)
            and len({(item.get("service_instance_id"), item.get("model_instance_id")) for item in warm_records}) == 1
        )
        cold_records = [item for item in records if item["variant"] == name and item["phase"] == "cold" and item.get("repeat") is not None]
        summary[name]["cold_reinitialization_observed"] = (
            len(cold_records) >= 3
            and all(item.get("identity_status") == "matched" and isinstance(item.get("model_init_count"), int) and item["model_init_count"] >= 1 for item in cold_records)
            and len({item.get("model_instance_id") for item in cold_records}) == len(cold_records)
        )
    required = [item for item in records if item.get("repeat") is not None]
    return {
        "schema_version": "ocr-rescue-benchmark-v1",
        "passed": bool(config.get("cold_reset_command")) and bool(required) and all(item.get("quality_passed") for item in required) and all(item.get("identity_status") == "matched" for item in required) and all(summary[name]["warm_model_reuse_proven"] and summary[name]["cold_reinitialization_observed"] for name in summary),
        "manifest_sha256": manifest_hash,
        "config_sha256": _sha256(config_path),
        "experimental_variable": config["experimental_variable"],
        "cold_status": "operator_reset_command_run_cold_state_unverified" if config.get("cold_reset_command") else "unavailable_no_cold_reset",
        "competing_workloads": config.get("competing_workloads", "unrecorded"),
        "host_observed": {"machine": platform.machine(), "cpu_count": os.cpu_count(), "gpu_device": None},
        "records": records,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.config, args.manifest, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OCR rescue benchmark: {'PASS' if report['passed'] else 'FAIL'}; report={args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
