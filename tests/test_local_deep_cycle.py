"""Tests for scripts/run_local_deep_cycle.py."""

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_local_deep_cycle as deep_cycle


class TestRunLocalDeepCycle:
    def test_create_deep_cycle_corpus(self, tmp_path):
        input_dir = tmp_path / "input"
        manifest = deep_cycle.create_deep_cycle_corpus(input_dir)

        assert input_dir.exists()
        assert set(manifest.keys()) == {
            "valid_pdf",
            "multi_tiff",
            "malformed_pdf",
            "invalid_jpg",
            "invalid_png",
        }
        for path in manifest.values():
            assert Path(path).exists()

    def test_summarize_outputs_without_failures_file(self, tmp_path):
        output_dir = tmp_path / "output"
        (output_dir / "EXPORT" / "PDF").mkdir(parents=True, exist_ok=True)
        (output_dir / "EXPORT" / "TEXT").mkdir(parents=True, exist_ok=True)

        (output_dir / "EXPORT" / "PDF" / "doc.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (output_dir / "EXPORT" / "TEXT" / "doc.txt").write_text("hello", encoding="utf-8")

        summary = deep_cycle.summarize_outputs(output_dir)

        assert summary["pdf_count"] == 1
        assert summary["text_count"] == 1
        assert summary["failures"]["rows"] == 0
        assert summary["failures"]["errors"] == {}

    def test_summarize_outputs_with_failures_csv(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        failures = output_dir / "failures.csv"
        failures.write_text(
            "Timestamp,File,Page,Error\n"
            "2026-02-24,file1.pdf,0,SOURCE_PAGE_COUNT_ZERO: pdf source reported 0 pages\n"
            "2026-02-24,file2.pdf,2,Missing chunk after terminal status RESUMED\n"
            "2026-02-24,file2.pdf,3,Missing chunk after terminal status RESUMED\n",
            encoding="utf-8",
        )

        summary = deep_cycle.summarize_outputs(output_dir)

        assert summary["failures"]["rows"] == 3
        assert summary["failures"]["errors"][
            "Missing chunk after terminal status RESUMED"
        ] == 2
        assert (
            summary["failures"]["errors"][
                "SOURCE_PAGE_COUNT_ZERO: pdf source reported 0 pages"
            ]
            == 1
        )

    def test_run_without_docker_writes_report_and_json(self, tmp_path):
        base_dir = tmp_path / "cycle"
        report = tmp_path / "report.md"
        json_output = tmp_path / "result.json"

        code = deep_cycle.run(
            base_dir=base_dir,
            run_docker=False,
            docker_image="unused",
            timeout_seconds=30,
            repo_root=tmp_path,
            use_worktree_code=True,
            report_path=report,
            json_output=json_output,
        )

        assert code == 0
        assert report.exists()
        assert json_output.exists()

        payload = json.loads(json_output.read_text(encoding="utf-8"))
        assert payload["docker"]["executed"] is False
        assert payload["docker"]["use_worktree_code"] is True
        assert payload["summary"]["failures"]["rows"] == 0

    def test_run_docker_cycle_command_includes_output_bound_failure_paths(
        self, tmp_path,
    ):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        repo_root = tmp_path / "repo"
        for folder in (input_dir, output_dir, temp_dir, repo_root):
            folder.mkdir(parents=True, exist_ok=True)

        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(deep_cycle.subprocess, "run", return_value=completed) as run_mock:
            deep_cycle.run_docker_cycle(
                image="ocr-gpu-local:latest",
                input_dir=input_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                timeout_seconds=10,
                repo_root=repo_root,
                use_worktree_code=True,
            )

        cmd = run_mock.call_args.args[0]
        cmd_text = " ".join(cmd)
        assert "FAILURE_REPORT=/app/deep_output/failures.csv" in cmd_text
        assert "LOG_DIR=/app/deep_output/logs" in cmd_text
        assert "ENABLE_VALIDATION=true" in cmd_text

    def test_analyze_runtime_output_detects_traceback_signatures(self):
        stdout = "INFO booting worker\n"
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"/app/ocr_gpu_async.py\", line 1, in <module>\n"
            "ModuleNotFoundError: No module named 'paddleocr'\n"
        )
        runtime = deep_cycle.analyze_runtime_output(stdout, stderr)

        assert runtime["total"] >= 2
        assert runtime["counts"]["traceback"] >= 1
        assert runtime["counts"]["module_not_found"] >= 1

    def test_run_docker_flags_unreported_runtime_signatures_as_anomaly(self, tmp_path):
        base_dir = tmp_path / "cycle"
        report = tmp_path / "report.md"
        json_output = tmp_path / "result.json"
        mocked = subprocess.CompletedProcess(
            args=["docker", "run"],
            returncode=0,
            stdout="starting worker\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'paddleocr'\n"
            ),
        )

        with mock.patch.object(deep_cycle, "run_docker_cycle", return_value=mocked):
            code = deep_cycle.run(
                base_dir=base_dir,
                run_docker=True,
                docker_image="ocr-gpu-local:latest",
                timeout_seconds=30,
                repo_root=tmp_path,
                use_worktree_code=True,
                report_path=report,
                json_output=json_output,
            )

        assert code == 2
        payload = json.loads(json_output.read_text(encoding="utf-8"))
        assert payload["summary"]["runtime_signatures"]["total"] >= 1
        assert payload["summary"]["runtime_signatures"]["counts"]["module_not_found"] >= 1
        assert payload["summary"]["anomalies"]
