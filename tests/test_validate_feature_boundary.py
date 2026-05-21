"""Tests for scripts/validate_feature_boundary.py."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_feature_boundary import (
    CheckResult,
    ValidationReport,
    check_ai_feature_defaults,
    check_api_reference_boundary,
    check_boundary_contract_doc,
    check_core_guardrails,
    check_job_submit_defaults,
    check_queue_isolation,
    format_json,
    format_markdown,
    format_text,
    main,
    run_all_checks,
)


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    api = tmp_path / "api"
    api.mkdir()
    (api / "models.py").write_text(
        textwrap.dedent("""\
            class JobSubmitRequest:
                enable_docintel: bool = False
                docintel_mode: str = "full"
                skip_ocr: bool = Field(
                    False,
                    description="Skip the primary OCR engine step. Only supported if the document is a native PDF with embedded text or if relying entirely on NLP extraction.",
                )
        """),
        encoding="utf-8",
    )
    (api / "deps.py").write_text(
        'skip_ocr: bool = Form(False, description="Skip OCR processing and only perform NLP/DocIntel if enabled. Assumes input is already textual.")\n',
        encoding="utf-8",
    )

    docs = tmp_path / "docs"
    (docs / "architecture").mkdir(parents=True)
    (docs / "06-CONFIGURATION-REFERENCE.md").write_text(
        textwrap.dedent("""\
            # Config
            | `ENABLE_VALIDATION` | `true` | Validation |
            | `API_AUDIT_LOG_ENABLED` | `true` | Audit |
            | `ENABLE_NER` | `false` | NER |
            | `ENABLE_CLASSIFICATION` | `false` | Classification |
            | `ENABLE_EXTRACTION` | `false` | Extraction |
            | `ENABLE_SPECIALIST_ROUTING` | `false` | Routing |

            ## Forensic-Core Defaults
            core defaults here

            ## AI-Adjacent Controls
            ai controls here
            See [docs/architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).
            Distributed queue isolation keeps advanced AI-adjacent work off the default OCR path.
        """),
        encoding="utf-8",
    )
    (docs / "07-TRANSFORMS-STAMPING.md").write_text(
        textwrap.dedent("""\
            # Transforms
            ## Scope Boundary
            Includes forensic safeguards and custody diagnostics.
        """),
        encoding="utf-8",
    )
    (docs / "API-REFERENCE.md").write_text(
        textwrap.dedent("""\
            # API
            > [!NOTE]
            > `enable_docintel` and `skip_ocr` are AI-adjacent controls.
            default forensic-core OCR path
            These controls add optional semantic or native-text behavior
            As with single-job submission, DocIntel settings are optional AI-adjacent enrichment controls rather than part of the minimum forensic-core processing guarantee.
            These endpoints are AI-adjacent analyst-assist features.
            [Forensic-Core vs AI-Adjacent Boundary](architecture/forensic-ai-boundary-contract.md)
            See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).
            See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).
            See [architecture/forensic-ai-boundary-contract.md](architecture/forensic-ai-boundary-contract.md).
        """),
        encoding="utf-8",
    )
    (docs / "architecture" / "forensic-ai-boundary-contract.md").write_text(
        textwrap.dedent("""\
            # Contract
            ## 2. Forensic-Core Scope
            ## 3. AI-Adjacent Scope
            ## 4. Current Enforcement Points
            Current Capability Map
            Contract Rules
            ## 6. Change Checklist
            Use scripts/validate_feature_boundary.py
        """),
        encoding="utf-8",
    )

    coordinator = tmp_path / "coordinator" / "coordinator"
    coordinator.mkdir(parents=True)
    (coordinator / "celery.py").write_text(
        textwrap.dedent("""\
            _STATIC_ROUTES = {
                'jobs.tasks.extract_structured_data': 'nlp_general',
                'jobs.tasks.process_text_only': 'cpu_general',
                'jobs.tasks_layoutlm.run_layoutlm_extraction': 'ocr_layoutlm',
            }
            app.conf.task_queues = [
                Queue('ocr_gpu'),
                Queue('cpu_general'),
                Queue('nlp_general'),
                Queue('ocr_layoutlm'),
            ]
        """),
        encoding="utf-8",
    )

    layoutlm = tmp_path / "coordinator" / "jobs"
    layoutlm.mkdir(parents=True)
    (layoutlm / "layoutlm_config.py").write_text(
        'ENABLE_LAYOUTLM: bool = os.environ.get("ENABLE_LAYOUTLM", "false").lower() in ("1", "true", "yes")\n',
        encoding="utf-8",
    )

    return tmp_path


class TestJobSubmitDefaults:
    def test_passes_on_expected_defaults(self, fake_project: Path) -> None:
        assert check_job_submit_defaults(fake_project).passed

    def test_fails_when_enable_docintel_default_changes(self, fake_project: Path) -> None:
        (fake_project / "api" / "models.py").write_text("enable_docintel: bool = True\n", encoding="utf-8")
        assert not check_job_submit_defaults(fake_project).passed


class TestCoreGuardrails:
    def test_passes_when_core_guardrails_present(self, fake_project: Path) -> None:
        assert check_core_guardrails(fake_project).passed

    def test_fails_when_scope_boundary_removed(self, fake_project: Path) -> None:
        (fake_project / "docs" / "07-TRANSFORMS-STAMPING.md").write_text("# Transforms\n", encoding="utf-8")
        assert not check_core_guardrails(fake_project).passed


class TestAiFeatureDefaults:
    def test_passes_when_ai_defaults_are_off(self, fake_project: Path) -> None:
        assert check_ai_feature_defaults(fake_project).passed

    def test_fails_when_layoutlm_default_changes(self, fake_project: Path) -> None:
        (fake_project / "coordinator" / "jobs" / "layoutlm_config.py").write_text(
            'ENABLE_LAYOUTLM = os.environ.get("ENABLE_LAYOUTLM", "true")\n',
            encoding="utf-8",
        )
        assert not check_ai_feature_defaults(fake_project).passed


class TestQueueIsolation:
    def test_passes_on_expected_routes(self, fake_project: Path) -> None:
        assert check_queue_isolation(fake_project).passed

    def test_fails_when_layoutlm_route_missing(self, fake_project: Path) -> None:
        (fake_project / "coordinator" / "coordinator" / "celery.py").write_text(
            "'jobs.tasks.extract_structured_data': 'nlp_general'\n",
            encoding="utf-8",
        )
        assert not check_queue_isolation(fake_project).passed


class TestBoundaryContractDoc:
    def test_passes_with_required_sections(self, fake_project: Path) -> None:
        assert check_boundary_contract_doc(fake_project).passed

    def test_fails_when_checklist_missing(self, fake_project: Path) -> None:
        (fake_project / "docs" / "architecture" / "forensic-ai-boundary-contract.md").write_text(
            "## 2. Forensic-Core Scope\n",
            encoding="utf-8",
        )
        assert not check_boundary_contract_doc(fake_project).passed

    def test_fails_when_legacy_contract_still_exists(self, fake_project: Path) -> None:
        (fake_project / "docs" / "architecture" / "forensic-ai-boundary.md").write_text(
            "# legacy\n",
            encoding="utf-8",
        )
        assert not check_boundary_contract_doc(fake_project).passed


class TestApiReferenceBoundary:
    def test_passes_with_boundary_note(self, fake_project: Path) -> None:
        assert check_api_reference_boundary(fake_project).passed

    def test_fails_when_api_note_missing(self, fake_project: Path) -> None:
        (fake_project / "docs" / "API-REFERENCE.md").write_text("# API\n", encoding="utf-8")
        assert not check_api_reference_boundary(fake_project).passed


class TestFormattingAndRunner:
    def test_run_all_checks(self, fake_project: Path) -> None:
        report = run_all_checks(fake_project)
        assert report.passed
        assert len(report.checks) == 6

    def test_format_text(self, fake_project: Path) -> None:
        assert "Feature Boundary Validation Report" in format_text(run_all_checks(fake_project))

    def test_format_json(self, fake_project: Path) -> None:
        data = json.loads(format_json(run_all_checks(fake_project)))
        assert data["summary"]["total"] == 6

    def test_format_markdown(self, fake_project: Path) -> None:
        assert "# Feature Boundary Validation Report" in format_markdown(run_all_checks(fake_project))

    def test_main_exit_zero_on_pass(self, fake_project: Path) -> None:
        assert main(["--project-root", str(fake_project)]) == 0

    def test_main_exit_one_on_fail(self, fake_project: Path) -> None:
        (fake_project / "docs" / "API-REFERENCE.md").write_text("# API\n", encoding="utf-8")
        assert main(["--project-root", str(fake_project)]) == 1

    def test_main_json_output(self, fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--project-root", str(fake_project), "--json"])
        assert code == 0
        assert json.loads(capsys.readouterr().out)["passed"] is True


class TestValidationReport:
    def test_failing_check_marks_report_failed(self) -> None:
        report = ValidationReport()
        report.add(CheckResult(name="x", passed=False, message="bad"))
        assert not report.passed
