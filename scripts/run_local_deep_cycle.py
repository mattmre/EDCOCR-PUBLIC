"""Run a reproducible local deep-cycle OCR corpus and summarize outputs.

This script is designed for local/docker reliability rehearsal. It can:
1) Create a mixed corpus (valid + malformed inputs),
2) Optionally execute OCR in Docker against isolated mount points,
3) Summarize outputs/failures and write markdown/json evidence artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

SIDECAR_EXPORT_DIRS = [
    "VALIDATION",
    "NER",
    "HANDWRITING",
    "CLASSIFICATION",
    "EXTRACTION",
]

RUNTIME_SIGNATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "module_not_found": re.compile(r"ModuleNotFoundError:"),
    "import_error": re.compile(r"ImportError:"),
    "segfault": re.compile(r"Segmentation fault|core dumped", re.IGNORECASE),
    "killed": re.compile(r"(^|\\s)Killed($|\\s)", re.IGNORECASE),
    "cuda_oom": re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
}


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def create_deep_cycle_corpus(input_dir: Path) -> dict[str, str]:
    """Create a deterministic mixed corpus for deep-cycle validation."""
    input_dir.mkdir(parents=True, exist_ok=True)

    valid_pdf = input_dir / "valid_4page.pdf"
    _write_valid_pdf(valid_pdf, pages=4)

    multi_tiff = input_dir / "multi_frame.tiff"
    _write_multi_frame_tiff(multi_tiff, frames=3)

    malformed_pdf = input_dir / "malformed_zero_page.pdf"
    malformed_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    invalid_jpg = input_dir / "invalid_sample.jpg"
    invalid_jpg.write_bytes(b"not-a-jpeg")

    invalid_png = input_dir / "invalid_sample.png"
    invalid_png.write_bytes(b"not-a-png")

    return {
        "valid_pdf": str(valid_pdf),
        "multi_tiff": str(multi_tiff),
        "malformed_pdf": str(malformed_pdf),
        "invalid_jpg": str(invalid_jpg),
        "invalid_png": str(invalid_png),
    }


def _write_valid_pdf(path: Path, pages: int = 4) -> None:
    doc = fitz.open()
    for idx in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"deep-cycle test page {idx + 1}")
    doc.save(path)
    doc.close()


def _write_multi_frame_tiff(path: Path, frames: int = 3) -> None:
    base_frames: list[Image.Image] = []
    for idx in range(frames):
        frame = Image.new("RGB", (64, 64), color=(40 * idx, 10 * idx, 20 * idx))
        base_frames.append(frame)
    base_frames[0].save(path, save_all=True, append_images=base_frames[1:])


def run_docker_cycle(
    image: str,
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    timeout_seconds: int,
    repo_root: Path,
    use_worktree_code: bool,
) -> subprocess.CompletedProcess[str]:
    """Run OCR pipeline inside Docker against isolated deep-cycle mounts."""
    script_path = "/app/ocr_gpu_async.py"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "SOURCE_FOLDER=/app/deep_input",
        "-e",
        "OUTPUT_FOLDER=/app/deep_output",
        "-e",
        "TEMP_FOLDER=/app/deep_temp",
        "-e",
        "LOG_DIR=/app/deep_output/logs",
        "-e",
        "FAILURE_REPORT=/app/deep_output/failures.csv",
        "-e",
        "ENABLE_VALIDATION=true",
        "-e",
        "ENABLE_NER=true",
        "-e",
        "ENABLE_HANDWRITING=true",
        "-e",
        "ENABLE_CLASSIFICATION=true",
        "-e",
        "ENABLE_EXTRACTION=true",
        "-v",
        f"{input_dir.resolve()}:/app/deep_input",
        "-v",
        f"{output_dir.resolve()}:/app/deep_output",
        "-v",
        f"{temp_dir.resolve()}:/app/deep_temp",
    ]
    if use_worktree_code:
        script_path = "/workspace/ocr_gpu_async.py"
        cmd.extend(
            [
                "-v",
                f"{repo_root.resolve()}:/workspace",
                "-e",
                "PYTHONPATH=/workspace:/app",
            ],
        )
    cmd.extend([image, "python3", script_path])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def summarize_outputs(output_dir: Path) -> dict[str, Any]:
    """Summarize deep-cycle outputs and failure evidence."""
    export_dir = output_dir / "EXPORT"
    summary: dict[str, Any] = {
        "pdf_count": _count_files(export_dir / "PDF", "*.pdf"),
        "text_count": _count_files(export_dir / "TEXT", "*.txt"),
        "sidecar_dirs_present": {},
        "sidecar_file_counts": {},
        "failures": {
            "path": str(output_dir / "failures.csv"),
            "rows": 0,
            "errors": {},
        },
    }

    for folder in SIDECAR_EXPORT_DIRS:
        path = export_dir / folder
        present = path.exists()
        summary["sidecar_dirs_present"][folder] = present
        summary["sidecar_file_counts"][folder] = _count_files(path, "*.json")

    failures_path = output_dir / "failures.csv"
    if failures_path.exists():
        errors: Counter[str] = Counter()
        rows = 0
        with failures_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows += 1
                errors[row.get("Error", "UNKNOWN")] += 1
        summary["failures"] = {
            "path": str(failures_path),
            "rows": rows,
            "errors": dict(errors),
        }

    return summary


def analyze_runtime_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Classify high-risk runtime signatures from docker output streams."""
    counts: Counter[str] = Counter()
    matches: list[dict[str, str]] = []
    joined = "\n".join([(stdout or ""), (stderr or "")])

    for raw_line in joined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for signature, pattern in RUNTIME_SIGNATURE_PATTERNS.items():
            if not pattern.search(line):
                continue
            counts[signature] += 1
            if len(matches) < 50:
                matches.append({"signature": signature, "line": line})
            break

    return {
        "total": int(sum(counts.values())),
        "counts": dict(counts),
        "matches": matches,
    }


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def write_markdown_report(
    report_path: Path,
    manifest: dict[str, str],
    summary: dict[str, Any],
    docker_rc: int | None,
    docker_stdout_tail: str,
    docker_stderr_tail: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local Deep-Cycle Report",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Docker executed: {'yes' if docker_rc is not None else 'no'}",
    ]
    if docker_rc is not None:
        lines.append(f"- Docker return code: {docker_rc}")
    lines.extend(
        [
            "",
            "## Corpus Manifest",
            "",
        ]
    )
    for key, value in manifest.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Output Summary",
            "",
            f"- PDF count: {summary['pdf_count']}",
            f"- TEXT count: {summary['text_count']}",
            "",
            "### Sidecar Presence",
            "",
        ]
    )
    for folder in SIDECAR_EXPORT_DIRS:
        present = summary["sidecar_dirs_present"][folder]
        count = summary["sidecar_file_counts"][folder]
        lines.append(f"- {folder}: present={present}, files={count}")

    lines.extend(
        [
            "",
            "### Failures",
            "",
            f"- failures.csv: `{summary['failures']['path']}`",
            f"- rows: {summary['failures']['rows']}",
        ]
    )
    for error, count in summary["failures"]["errors"].items():
        lines.append(f"- {error}: {count}")

    runtime_signatures = summary.get("runtime_signatures", {"total": 0, "counts": {}, "matches": []})
    lines.extend(
        [
            "",
            "### Runtime Signatures",
            "",
            f"- total matches: {runtime_signatures.get('total', 0)}",
        ]
    )
    for signature, count in runtime_signatures.get("counts", {}).items():
        lines.append(f"- {signature}: {count}")
    sample_matches = runtime_signatures.get("matches", [])[:12]
    if sample_matches:
        lines.extend(["", "#### Runtime Signature Samples", ""])
        for item in sample_matches:
            lines.append(f"- [{item.get('signature', 'unknown')}] `{item.get('line', '')}`")

    anomalies = summary.get("anomalies", [])
    if anomalies:
        lines.extend(["", "### Anomalies", ""])
        for anomaly in anomalies:
            lines.append(f"- {anomaly}")

    if docker_stdout_tail:
        lines.extend(
            [
                "",
                "## Docker stdout tail",
                "",
                "```text",
                docker_stdout_tail,
                "```",
            ]
        )

    if docker_stderr_tail:
        lines.extend(
            [
                "",
                "## Docker stderr tail",
                "",
                "```text",
                docker_stderr_tail,
                "```",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    base_dir: Path,
    run_docker: bool,
    docker_image: str,
    timeout_seconds: int,
    repo_root: Path,
    use_worktree_code: bool,
    report_path: Path | None,
    json_output: Path | None,
) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    temp_dir = base_dir / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest = create_deep_cycle_corpus(input_dir)

    docker_result: subprocess.CompletedProcess[str] | None = None
    docker_stdout_tail = ""
    docker_stderr_tail = ""
    if run_docker:
        docker_result = run_docker_cycle(
            docker_image,
            input_dir=input_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            timeout_seconds=timeout_seconds,
            repo_root=repo_root,
            use_worktree_code=use_worktree_code,
        )
        if docker_result.stdout:
            docker_stdout_tail = "\n".join(docker_result.stdout.splitlines()[-80:])
        if docker_result.stderr:
            docker_stderr_tail = "\n".join(docker_result.stderr.splitlines()[-40:])

    summary = summarize_outputs(output_dir)
    summary["runtime_signatures"] = {"total": 0, "counts": {}, "matches": []}
    anomalies: list[str] = []
    if run_docker:
        runtime_signatures = analyze_runtime_output(
            docker_result.stdout if docker_result else "",
            docker_result.stderr if docker_result else "",
        )
        summary["runtime_signatures"] = runtime_signatures
        if runtime_signatures.get("total", 0) > 0:
            anomalies.append(
                "Detected traceback/import/runtime signatures in docker output. "
                "Review runtime signature section for transitive/dependency issues.",
            )
        no_outputs = summary["pdf_count"] == 0 and summary["text_count"] == 0
        no_failures = summary["failures"]["rows"] == 0
        if no_outputs and no_failures:
            anomalies.append(
                "Docker run produced no PDF/TEXT outputs and no failures.csv rows. "
                "Likely source-path/config mismatch or stale in-container code.",
            )
        if no_failures and runtime_signatures.get("total", 0) > 0:
            anomalies.append(
                "Runtime error signatures were detected, but failures.csv has 0 rows. "
                "Potential unreported/transitive failure path detected.",
            )
    summary["anomalies"] = anomalies
    result = {
        "base_dir": str(base_dir),
        "manifest": manifest,
        "summary": summary,
        "docker": {
            "executed": run_docker,
            "return_code": docker_result.returncode if docker_result else None,
            "use_worktree_code": use_worktree_code,
        },
    }

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if report_path:
        write_markdown_report(
            report_path=report_path,
            manifest=manifest,
            summary=summary,
            docker_rc=docker_result.returncode if docker_result else None,
            docker_stdout_tail=docker_stdout_tail,
            docker_stderr_tail=docker_stderr_tail,
        )

    if docker_result and docker_result.returncode != 0:
        return 1
    if anomalies:
        return 2
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local OCR deep-cycle rehearsal.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("tmp") / f"deep_cycle_{_ts()}",
        help="Base directory where input/output/temp folders are created.",
    )
    parser.add_argument(
        "--run-docker",
        action="store_true",
        help="Execute OCR in Docker after creating corpus.",
    )
    parser.add_argument(
        "--docker-image",
        default="ocr-gpu-local:latest",
        help="Docker image to execute when --run-docker is set.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Docker execution timeout in seconds.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root to mount when using worktree code execution.",
    )
    parser.add_argument(
        "--use-image-code",
        action="store_true",
        help="Use the image's bundled /app code instead of mounting worktree code.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown report output path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        base_dir=args.base_dir,
        run_docker=args.run_docker,
        docker_image=args.docker_image,
        timeout_seconds=args.timeout_seconds,
        repo_root=args.repo_root,
        use_worktree_code=not args.use_image_code,
        report_path=args.report,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
