from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_playwright_run as summarize  # noqa: E402


def _write_results(tmp_path: Path, *, expected: int, skipped: int, unexpected: int) -> Path:
    payload = {
        "config": {
            "metadata": {
                "runId": "PW-TEST-001",
                "artifactRoot": str(tmp_path / "artifacts" / "PW-TEST-001"),
            },
            "projects": [
                {"name": "request-api"},
                {"name": "ops-api"},
            ],
        },
        "suites": [
            {
                "title": "request/jobs.spec.js",
                "file": "request/jobs.spec.js",
                "specs": [
                    {
                        "title": "job request contract",
                        "tests": [
                            {
                                "status": "expected" if unexpected == 0 else "unexpected",
                                "annotations": [
                                    {
                                        "type": "skip",
                                        "description": "Set PLAYWRIGHT_API_BASE_URL to run request-level API lifecycle tests.",
                                    }
                                ]
                                if skipped
                                else [],
                                "results": [],
                            }
                        ],
                    }
                ],
                "suites": [],
            }
        ],
        "errors": [],
        "stats": {
            "duration": 1250,
            "expected": expected,
            "skipped": skipped,
            "unexpected": unexpected,
            "flaky": 0,
        },
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    return results_path


def test_summarizer_marks_all_skipped_run_blocked_and_updates_ledger(tmp_path: Path) -> None:
    results_path = _write_results(tmp_path, expected=0, skipped=1, unexpected=0)
    summary_md = tmp_path / "summary.md"
    summary_json = tmp_path / "summary.json"
    ledger_path = tmp_path / "ledger.md"
    ledger_path.write_text(
        "# Playwright Run Ledger\n\n| Run ID | Date | Scope | Branch/PR | Status | Summary Path |\n|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )

    exit_code = summarize.main(
        [
            "--results-json",
            str(results_path),
            "--summary-markdown",
            str(summary_md),
            "--summary-json",
            str(summary_json),
            "--ledger-path",
            str(ledger_path),
            "--ledger-summary-path",
            "docs/testing/runs/PW-TEST-001.md",
            "--trigger",
            "unit test",
            "--branch",
            "codex/test-branch",
            "--suite-scope",
            "phase9 validation",
        ]
    )

    assert exit_code == 0
    normalized = json.loads(summary_json.read_text(encoding="utf-8"))
    assert normalized["status"] == "BLOCKED"
    markdown = summary_md.read_text(encoding="utf-8")
    assert "Overall status: `BLOCKED`" in markdown
    ledger = ledger_path.read_text(encoding="utf-8")
    assert "PW-TEST-001" in ledger
    assert "docs/testing/runs/PW-TEST-001.md" in ledger


def test_summarizer_marks_unexpected_failures_fail(tmp_path: Path) -> None:
    results_path = _write_results(tmp_path, expected=0, skipped=0, unexpected=1)
    summary_md = tmp_path / "summary.md"
    summary_json = tmp_path / "summary.json"

    exit_code = summarize.main(
        [
            "--results-json",
            str(results_path),
            "--summary-markdown",
            str(summary_md),
            "--summary-json",
            str(summary_json),
            "--trigger",
            "unit test",
            "--branch",
            "codex/test-branch",
            "--suite-scope",
            "phase9 validation",
        ]
    )

    assert exit_code == 0
    normalized = json.loads(summary_json.read_text(encoding="utf-8"))
    assert normalized["status"] == "FAIL"
    assert normalized["failed_specs"] == ["job request contract"]
