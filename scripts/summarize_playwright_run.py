#!/usr/bin/env python3
"""Normalize Playwright JSON output into durable markdown and summary JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_specs(suites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for suite in suites:
        collected.extend(suite.get("specs", []))
        collected.extend(_collect_specs(suite.get("suites", [])))
    return collected


def _collect_suite_files(suites: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for suite in suites:
        suite_file = suite.get("file")
        if isinstance(suite_file, str) and suite_file:
            files.append(suite_file)
        files.extend(_collect_suite_files(suite.get("suites", [])))
    return files


def _collect_skip_reasons(specs: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for spec in specs:
        for test in spec.get("tests", []):
            for annotation in test.get("annotations", []):
                if annotation.get("type") == "skip" and annotation.get("description"):
                    reasons.append(str(annotation["description"]))
            for result in test.get("results", []):
                for annotation in result.get("annotations", []):
                    if annotation.get("type") == "skip" and annotation.get("description"):
                        reasons.append(str(annotation["description"]))
    counts = Counter(reasons)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [reason for reason, _count in ordered]


def _collect_failed_specs(specs: list[dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    for spec in specs:
        for test in spec.get("tests", []):
            if test.get("status") == "unexpected":
                failed.append(str(spec.get("title", "<untitled spec>")))
                break
    return failed


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _derive_status(stats: dict[str, Any], errors: list[Any], skip_reasons: list[str]) -> str:
    unexpected = int(stats.get("unexpected", 0) or 0)
    expected = int(stats.get("expected", 0) or 0)
    skipped = int(stats.get("skipped", 0) or 0)
    flaky = int(stats.get("flaky", 0) or 0)

    if unexpected > 0 or errors:
        return "FAIL"
    if expected == 0 and skipped > 0:
        return "BLOCKED" if skip_reasons else "PASS_WITH_FOLLOWUPS"
    if skipped > 0 or flaky > 0:
        return "PASS_WITH_FOLLOWUPS"
    return "PASS"


def _format_duration(duration_ms: float) -> str:
    seconds = max(duration_ms / 1000.0, 0.0)
    if seconds < 60:
        return f"~{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"~{minutes}m {remainder:.1f}s"


def _format_list(lines: list[str], items: list[str], fallback: str = "none") -> None:
    if items:
        for item in items:
            lines.append(f"- {item}")
    else:
        lines.append(f"- {fallback}")


def _artifact_paths(artifact_root: str | None) -> dict[str, str]:
    if not artifact_root:
        return {
            "html_report": "not available",
            "junit_xml": "not available",
            "json_summary": "not available",
            "blob_report": "not available",
            "test_results": "not available",
        }
    root = Path(artifact_root)
    return {
        "html_report": str(root / "html-report"),
        "junit_xml": str(root / "results.xml"),
        "json_summary": str(root / "results.json"),
        "blob_report": str(root / "blob-report"),
        "test_results": str(root / "test-results"),
    }


def _write_summary_markdown(
    output_path: Path,
    *,
    run_id: str,
    run_date: str,
    trigger: str,
    branch: str,
    commit: str,
    pr: str,
    operator: str,
    suite_scope: str,
    surfaces: list[str],
    feature_flags: list[str],
    environment_urls: list[str],
    auth_modes: list[str],
    status: str,
    passed: int,
    failed: int,
    skipped: int,
    duration: str,
    artifact_paths: dict[str, str],
    key_findings: list[str],
    known_blockers: list[str],
    deferred_items: list[str],
    follow_ups: list[str],
    notes: list[str],
) -> None:
    lines = [
        f"# Playwright Run - {run_id}",
        "",
        "## Run Metadata",
        "",
        f"- Run ID: `{run_id}`",
        f"- Date: `{run_date}`",
        f"- Trigger: `{trigger}`",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- PR: `{pr}`",
        f"- Operator: `{operator}`",
        "",
        "## Scope",
        "",
        f"- Suite scope: `{suite_scope}`",
        "Surfaces covered:",
    ]
    _format_list(lines, surfaces, fallback="derived from Playwright suite inventory")
    lines.extend([
        "Feature flags:",
    ])
    _format_list(lines, feature_flags)
    lines.extend([
        "Environment URLs:",
    ])
    _format_list(lines, environment_urls)
    lines.extend([
        "Auth mode:",
    ])
    _format_list(lines, auth_modes)
    lines.extend([
        "",
        "## Results Summary",
        "",
        f"- Overall status: `{status}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failed}`",
        f"- Skipped: `{skipped}`",
        f"- Duration: `{duration}`",
        "",
        "## Artifact Locations",
        "",
        f"- HTML report: `{artifact_paths['html_report']}`",
        f"- JUnit XML: `{artifact_paths['junit_xml']}`",
        f"- JSON summary: `{artifact_paths['json_summary']}`",
        f"- Blob report: `{artifact_paths['blob_report']}`",
        f"- Trace/video/screenshot folder: `{artifact_paths['test_results']}`",
        "",
        "## Key Findings",
        "",
    ])
    _format_list(lines, key_findings)
    lines.extend([
        "",
        "## PR Assessment",
        "",
        f"- Merge recommendation: `{'review failures before merge' if status == 'FAIL' else 'informative gate ready for review'}`",
        "Known blockers:",
    ])
    _format_list(lines, known_blockers)
    lines.extend([
        "Deferred items:",
    ])
    _format_list(lines, deferred_items)
    lines.extend([
        "",
        "## Follow-Up Test Queue",
        "",
    ])
    _format_list(lines, follow_ups)
    lines.extend([
        "",
        "## Notes For Next Run",
        "",
    ])
    _format_list(lines, notes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_ledger_row(
    ledger_path: Path,
    *,
    run_id: str,
    run_date: str,
    suite_scope: str,
    branch: str,
    pr: str,
    status: str,
    summary_path: str,
) -> None:
    existing = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
    if run_id in existing:
        return
    branch_or_pr = branch if pr in {"", "pending", "n/a"} else f"{branch} / {pr}"
    row = f"| {run_id} | {run_date} | {suite_scope} | `{branch_or_pr}` | {status} | `{summary_path}` |"
    lines = existing.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("|---"):
            insert_at = index + 1
            break
    if insert_at:
        lines.insert(insert_at, row)
        content = "\n".join(lines).rstrip() + "\n"
    else:
        content = existing.rstrip() + "\n" + row + "\n"
    ledger_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--summary-markdown", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--ledger-summary-path", type=str)
    parser.add_argument("--trigger", default="Playwright run")
    parser.add_argument("--branch", default="unknown")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--pr", default="pending")
    parser.add_argument("--operator", default="Codex")
    parser.add_argument("--suite-scope", default="Playwright suite")
    parser.add_argument("--surface", action="append", default=[])
    parser.add_argument("--feature-flag", action="append", default=[])
    parser.add_argument("--environment-url", action="append", default=[])
    parser.add_argument("--auth-mode", action="append", default=[])
    parser.add_argument("--known-blocker", action="append", default=[])
    parser.add_argument("--deferred-item", action="append", default=[])
    parser.add_argument("--follow-up", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _load_results(args.results_json)
    config = payload.get("config", {})
    stats = payload.get("stats", {})
    suites = payload.get("suites", [])
    errors = payload.get("errors", [])
    metadata = config.get("metadata", {})

    specs = _collect_specs(suites)
    derived_surfaces = [f"`{path}`" for path in _dedupe(_collect_suite_files(suites))]
    skip_reasons = _collect_skip_reasons(specs)
    failed_specs = _collect_failed_specs(specs)
    status = _derive_status(stats, errors, skip_reasons)
    run_id = str(metadata.get("runId") or args.summary_markdown.stem)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    artifact_paths = _artifact_paths(metadata.get("artifactRoot"))

    passed = int(stats.get("expected", 0) or 0)
    failed = int(stats.get("unexpected", 0) or 0)
    skipped = int(stats.get("skipped", 0) or 0)
    flaky = int(stats.get("flaky", 0) or 0)
    duration = _format_duration(float(stats.get("duration", 0.0) or 0.0))
    project_names = [
        str(project.get("name"))
        for project in config.get("projects", [])
        if project.get("name")
    ]

    key_findings: list[str] = []
    if status == "BLOCKED":
        key_findings.append(
            "The suite produced no executed assertions; the run is blocked on environment or credential prerequisites."
        )
    elif status == "FAIL":
        key_findings.append("At least one Playwright assertion failed and requires review before merge.")
    else:
        key_findings.append("The Playwright run completed and produced normalized reporting artifacts.")
    if project_names:
        key_findings.append(
            f"Projects observed in this run: {', '.join(project_names)}."
        )
    if flaky:
        key_findings.append(f"The run reported `{flaky}` flaky test result(s).")
    if skip_reasons:
        key_findings.append(
            "Top skip reasons: " + "; ".join(skip_reasons[:3]) + "."
        )
    if failed_specs:
        key_findings.append(
            "Failed specs: " + ", ".join(f"`{name}`" for name in failed_specs[:5]) + "."
        )

    known_blockers = list(args.known_blocker)
    if status == "BLOCKED":
        known_blockers.extend(skip_reasons[:5])
    if status == "FAIL":
        known_blockers.extend(failed_specs[:5])

    notes = list(args.note)
    if not notes:
        notes.append(f"Projects in scope: {', '.join(project_names) if project_names else 'unknown'}")
    if skip_reasons:
        notes.append("Skip posture: " + "; ".join(skip_reasons[:3]))

    normalized_summary = {
        "run_id": run_id,
        "date": run_date,
        "trigger": args.trigger,
        "branch": args.branch,
        "commit": args.commit,
        "pr": args.pr,
        "operator": args.operator,
        "suite_scope": args.suite_scope,
        "status": status,
        "counts": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "flaky": flaky,
        },
        "duration": duration,
        "projects": project_names,
        "surfaces": args.surface or derived_surfaces,
        "feature_flags": args.feature_flag,
        "environment_urls": args.environment_url,
        "auth_modes": args.auth_mode,
        "artifact_paths": artifact_paths,
        "skip_reasons": skip_reasons,
        "failed_specs": failed_specs,
        "errors": errors,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(normalized_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_summary_markdown(
        args.summary_markdown,
        run_id=run_id,
        run_date=run_date,
        trigger=args.trigger,
        branch=args.branch,
        commit=args.commit,
        pr=args.pr,
        operator=args.operator,
        suite_scope=args.suite_scope,
        surfaces=args.surface or derived_surfaces,
        feature_flags=args.feature_flag,
        environment_urls=args.environment_url,
        auth_modes=args.auth_mode,
        status=status,
        passed=passed,
        failed=failed,
        skipped=skipped,
        duration=duration,
        artifact_paths=artifact_paths,
        key_findings=key_findings,
        known_blockers=_dedupe(known_blockers),
        deferred_items=_dedupe(args.deferred_item),
        follow_ups=_dedupe(args.follow_up),
        notes=_dedupe(notes),
    )

    if args.ledger_path:
        _append_ledger_row(
            args.ledger_path,
            run_id=run_id,
            run_date=run_date,
            suite_scope=args.suite_scope,
            branch=args.branch,
            pr=args.pr,
            status=status,
            summary_path=args.ledger_summary_path or str(args.summary_markdown),
        )

    print(f"Playwright summary status: {status}")
    print(f"Markdown: {args.summary_markdown}")
    print(f"JSON: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
