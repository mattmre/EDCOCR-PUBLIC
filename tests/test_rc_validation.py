"""Tests for scripts/rc_validation.py.

Covers GateStatus enum, GateResult/RCReport dataclasses, each individual
gate function, run_validation integration, and report output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rc_validation import (
    GateResult,
    GateStatus,
    RCReport,
    _find_project_root,
    gate_api_compatibility,
    gate_changelog_check,
    gate_clean_worktree,
    gate_docs_check,
    gate_lint_check,
    gate_security_scan,
    gate_test_suite,
    gate_version_check,
    print_report,
    run_validation,
)

# ---------------------------------------------------------------------------
# GateStatus enum
# ---------------------------------------------------------------------------


class TestGateStatus:
    """Tests for the GateStatus enum."""

    def test_values(self):
        assert GateStatus.PASS.value == "pass"
        assert GateStatus.FAIL.value == "fail"
        assert GateStatus.SKIP.value == "skip"
        assert GateStatus.WARN.value == "warn"

    def test_member_count(self):
        assert len(GateStatus) == 4


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------


class TestGateResult:
    """Tests for the GateResult dataclass."""

    def test_creation(self):
        r = GateResult("RC-001", "Test gate", GateStatus.PASS, "OK")
        assert r.gate_id == "RC-001"
        assert r.name == "Test gate"
        assert r.status == GateStatus.PASS
        assert r.message == "OK"
        assert r.duration_seconds == 0.0
        assert r.details == ""

    def test_with_all_fields(self):
        r = GateResult("RC-002", "Full gate", GateStatus.FAIL, "Bad", 1.5, "detail")
        assert r.duration_seconds == 1.5
        assert r.details == "detail"


# ---------------------------------------------------------------------------
# RCReport dataclass
# ---------------------------------------------------------------------------


class TestRCReport:
    """Tests for the RCReport dataclass."""

    def test_empty_report(self):
        r = RCReport()
        assert r.passed == 0
        assert r.failed == 0
        assert r.skipped == 0
        assert r.warned == 0
        assert r.gates == []
        assert r.overall_status == GateStatus.PASS

    def test_add_pass(self):
        r = RCReport()
        r.add(GateResult("T-1", "pass", GateStatus.PASS, "ok"))
        assert r.passed == 1
        assert r.failed == 0
        assert len(r.gates) == 1
        assert r.overall_status == GateStatus.PASS

    def test_add_fail(self):
        r = RCReport()
        r.add(GateResult("T-2", "fail", GateStatus.FAIL, "bad"))
        assert r.passed == 0
        assert r.failed == 1
        assert r.overall_status == GateStatus.FAIL

    def test_add_skip(self):
        r = RCReport()
        r.add(GateResult("T-3", "skip", GateStatus.SKIP, "skipped"))
        assert r.skipped == 1
        assert r.overall_status == GateStatus.PASS

    def test_add_warn(self):
        r = RCReport()
        r.add(GateResult("T-4", "warn", GateStatus.WARN, "warning"))
        assert r.warned == 1
        assert r.overall_status == GateStatus.WARN

    def test_overall_pass_to_fail_propagation(self):
        r = RCReport()
        r.add(GateResult("T-1", "pass", GateStatus.PASS, "ok"))
        assert r.overall_status == GateStatus.PASS
        r.add(GateResult("T-2", "fail", GateStatus.FAIL, "bad"))
        assert r.overall_status == GateStatus.FAIL

    def test_overall_pass_to_warn_propagation(self):
        r = RCReport()
        r.add(GateResult("T-1", "pass", GateStatus.PASS, "ok"))
        assert r.overall_status == GateStatus.PASS
        r.add(GateResult("T-2", "warn", GateStatus.WARN, "warning"))
        assert r.overall_status == GateStatus.WARN

    def test_fail_not_downgraded_by_warn(self):
        r = RCReport()
        r.add(GateResult("T-1", "fail", GateStatus.FAIL, "bad"))
        r.add(GateResult("T-2", "warn", GateStatus.WARN, "warning"))
        assert r.overall_status == GateStatus.FAIL

    def test_mixed_counts(self):
        r = RCReport()
        r.add(GateResult("T-1", "p1", GateStatus.PASS, ""))
        r.add(GateResult("T-2", "p2", GateStatus.PASS, ""))
        r.add(GateResult("T-3", "f1", GateStatus.FAIL, ""))
        r.add(GateResult("T-4", "s1", GateStatus.SKIP, ""))
        r.add(GateResult("T-5", "w1", GateStatus.WARN, ""))
        assert r.passed == 2
        assert r.failed == 1
        assert r.skipped == 1
        assert r.warned == 1

    def test_to_dict_structure(self):
        r = RCReport(version="1.0.0", timestamp="2025-01-01T00:00:00")
        r.add(GateResult("RC-001", "Version check", GateStatus.PASS, "Version: 1.0.0", 0.1))
        d = r.to_dict()
        assert d["version"] == "1.0.0"
        assert d["timestamp"] == "2025-01-01T00:00:00"
        assert d["overall_status"] == "pass"
        assert d["summary"]["passed"] == 1
        assert d["summary"]["failed"] == 0
        assert d["summary"]["skipped"] == 0
        assert d["summary"]["warned"] == 0
        assert d["summary"]["total"] == 1
        assert len(d["gates"]) == 1
        g = d["gates"][0]
        assert g["gate_id"] == "RC-001"
        assert g["status"] == "pass"
        assert g["duration_seconds"] == 0.1

    def test_to_dict_json_serializable(self):
        r = RCReport(version="0.9.0")
        r.add(GateResult("RC-001", "test", GateStatus.FAIL, "msg"))
        # Should not raise
        output = json.dumps(r.to_dict())
        assert "RC-001" in output


# ---------------------------------------------------------------------------
# gate_version_check
# ---------------------------------------------------------------------------


class TestGateVersionCheck:
    """Tests for gate_version_check."""

    def test_valid_version(self, tmp_path):
        vf = tmp_path / "version.py"
        vf.write_text('__version__ = "1.0.0"\n')
        result = gate_version_check(tmp_path)
        assert result.status == GateStatus.PASS
        assert result.gate_id == "RC-001"
        assert "1.0.0" in result.message

    def test_missing_version_file(self, tmp_path):
        result = gate_version_check(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "not found" in result.message

    def test_invalid_version_content(self, tmp_path):
        vf = tmp_path / "version.py"
        vf.write_text('__version__ = "not-a-semver"\n')
        result = gate_version_check(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "No valid semver" in result.message

    def test_single_quoted_version(self, tmp_path):
        vf = tmp_path / "version.py"
        vf.write_text("__version__ = '2.3.4'\n")
        result = gate_version_check(tmp_path)
        assert result.status == GateStatus.PASS
        assert "2.3.4" in result.message


# ---------------------------------------------------------------------------
# gate_changelog_check
# ---------------------------------------------------------------------------


class TestGateChangelogCheck:
    """Tests for gate_changelog_check."""

    def test_present_and_long(self, tmp_path):
        cl = tmp_path / "CHANGELOG.md"
        cl.write_text("# Changelog\n\n" + "- entry\n" * 50)
        result = gate_changelog_check(tmp_path)
        assert result.status == GateStatus.PASS
        assert result.gate_id == "RC-002"

    def test_missing(self, tmp_path):
        result = gate_changelog_check(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "not found" in result.message

    def test_sparse_content(self, tmp_path):
        cl = tmp_path / "CHANGELOG.md"
        cl.write_text("# Changelog\n")
        result = gate_changelog_check(tmp_path)
        assert result.status == GateStatus.WARN
        assert "sparse" in result.message


# ---------------------------------------------------------------------------
# gate_docs_check
# ---------------------------------------------------------------------------


class TestGateDocsCheck:
    """Tests for gate_docs_check."""

    def test_all_docs_present(self, tmp_path):
        required = [
            "README.md",
            "CHANGELOG.md",
            "SECURITY.md",
            "LICENSE",
            "docs/api-stability-contract.md",
            "docs/migration-guide-v1.0.md",
            "docs/security-audit-checklist.md",
            "docs/compliance/soc2-readiness.md",
            "docs/compliance/hipaa-readiness.md",
        ]
        for doc in required:
            p = tmp_path / doc
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# {doc}\n")

        result = gate_docs_check(tmp_path)
        assert result.status == GateStatus.PASS
        assert result.gate_id == "RC-007"
        assert "9" in result.message

    def test_missing_docs(self, tmp_path):
        # Create only some docs
        (tmp_path / "README.md").write_text("# README\n")
        (tmp_path / "LICENSE").write_text("MIT\n")

        result = gate_docs_check(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "Missing" in result.message
        assert "CHANGELOG.md" in result.message


# ---------------------------------------------------------------------------
# gate_clean_worktree
# ---------------------------------------------------------------------------


class TestGateCleanWorktree:
    """Tests for gate_clean_worktree."""

    def test_clean_worktree(self, tmp_path):
        with mock.patch("rc_validation._run_command", return_value=(0, "", "", 0.1)):
            result = gate_clean_worktree(tmp_path)
        assert result.status == GateStatus.PASS
        assert result.gate_id == "RC-008"

    def test_dirty_worktree(self, tmp_path):
        dirty_output = " M file1.py\n?? file2.py\n"
        with mock.patch("rc_validation._run_command", return_value=(0, dirty_output, "", 0.1)):
            result = gate_clean_worktree(tmp_path)
        assert result.status == GateStatus.WARN
        assert "2 uncommitted" in result.message

    def test_git_not_available(self, tmp_path):
        with mock.patch("rc_validation._run_command", return_value=(-1, "", "git not found", 0.0)):
            result = gate_clean_worktree(tmp_path)
        assert result.status == GateStatus.WARN
        assert "Could not check" in result.message


# ---------------------------------------------------------------------------
# gate_api_compatibility
# ---------------------------------------------------------------------------


class TestGateApiCompatibility:
    """Tests for gate_api_compatibility."""

    def test_skip_when_no_versioning(self, tmp_path):
        result = gate_api_compatibility(tmp_path)
        assert result.status == GateStatus.SKIP
        assert "not found" in result.message

    def test_with_versioning_module(self, tmp_path):
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "__init__.py").write_text("")
        (api_dir / "versioning.py").write_text(
            "from enum import Enum\n"
            "class StabilityTier(Enum):\n"
            "    STABLE = 'stable'\n"
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class EndpointRecord:\n"
            "    method: str\n"
            "    path: str\n"
            "    name: str\n"
            "    tier: StabilityTier\n"
            "API_SURFACE = (\n"
            "    EndpointRecord('GET', '/api/v1/health', 'health', StabilityTier.STABLE),\n"
            ")\n"
            "def get_stable_endpoints():\n"
            "    return [e for e in API_SURFACE if e.tier == StabilityTier.STABLE]\n"
        )
        result = gate_api_compatibility(tmp_path)
        assert result.status == GateStatus.PASS
        assert "1 stable" in result.message


# ---------------------------------------------------------------------------
# gate_lint_check
# ---------------------------------------------------------------------------


class TestGateLintCheck:
    """Tests for gate_lint_check."""

    def test_lint_pass(self, tmp_path):
        with mock.patch("rc_validation._run_command", return_value=(0, "", "", 0.5)):
            result = gate_lint_check(tmp_path)
        assert result.status == GateStatus.PASS
        assert result.gate_id == "RC-003"
        assert result.duration_seconds == 0.5

    def test_lint_fail(self, tmp_path):
        stdout = "file.py:1:1 E501 line too long\nFound 1 error.\n"
        with mock.patch("rc_validation._run_command", return_value=(1, stdout, "", 0.3)):
            result = gate_lint_check(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "Found 1 error" in result.message


# ---------------------------------------------------------------------------
# gate_test_suite
# ---------------------------------------------------------------------------


class TestGateTestSuite:
    """Tests for gate_test_suite."""

    def test_skip_flag(self, tmp_path):
        result = gate_test_suite(tmp_path, skip=True)
        assert result.status == GateStatus.SKIP
        assert result.gate_id == "RC-004"
        assert "Skipped" in result.message

    def test_tests_pass(self, tmp_path):
        stdout = "42 passed in 5.3s\n"
        with mock.patch("rc_validation._run_command", return_value=(0, stdout, "", 5.3)):
            result = gate_test_suite(tmp_path)
        assert result.status == GateStatus.PASS
        assert "42 passed" in result.message

    def test_tests_fail(self, tmp_path):
        stdout = "40 passed, 2 failed in 6.0s\n"
        with mock.patch("rc_validation._run_command", return_value=(1, stdout, "", 6.0)):
            result = gate_test_suite(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "40 passed" in result.message
        assert "2 failed" in result.message


# ---------------------------------------------------------------------------
# gate_security_scan
# ---------------------------------------------------------------------------


class TestGateSecurityScan:
    """Tests for gate_security_scan."""

    def test_scanner_not_found(self, tmp_path):
        result = gate_security_scan(tmp_path)
        assert result.status == GateStatus.SKIP
        assert "not found" in result.message
        assert result.gate_id == "RC-005"

    def test_scan_no_findings(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "security_scan.py").write_text("# placeholder\n")
        scan_output = json.dumps({
            "summary": {"passed": 5, "failed": 0, "total": 5},
            "findings": [],
        })
        with mock.patch("rc_validation._run_command", return_value=(0, scan_output, "", 1.0)):
            result = gate_security_scan(tmp_path)
        assert result.status == GateStatus.PASS
        assert "No findings" in result.message

    def test_scan_critical_finding(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "security_scan.py").write_text("# placeholder\n")
        scan_output = json.dumps({
            "summary": {"passed": 4, "failed": 1, "total": 5},
            "findings": [
                {"check_id": "SEC-001", "passed": False, "severity": "critical"},
            ],
        })
        with mock.patch("rc_validation._run_command", return_value=(0, scan_output, "", 1.0)):
            result = gate_security_scan(tmp_path)
        assert result.status == GateStatus.FAIL
        assert "1 critical" in result.message

    def test_scan_non_critical_warning(self, tmp_path):
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "security_scan.py").write_text("# placeholder\n")
        scan_output = json.dumps({
            "summary": {"passed": 4, "failed": 1, "total": 5},
            "findings": [
                {"check_id": "SEC-002", "passed": False, "severity": "medium"},
            ],
        })
        with mock.patch("rc_validation._run_command", return_value=(0, scan_output, "", 1.0)):
            result = gate_security_scan(tmp_path)
        assert result.status == GateStatus.WARN
        assert "1 non-critical" in result.message


# ---------------------------------------------------------------------------
# run_validation integration
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Tests for the run_validation integration function."""

    def test_returns_rc_report(self, tmp_path):
        # Minimal project layout
        (tmp_path / "version.py").write_text('__version__ = "0.9.0"\n')
        (tmp_path / "CHANGELOG.md").write_text("# Changelog\n" + "- item\n" * 50)

        with mock.patch("rc_validation._run_command", return_value=(0, "", "", 0.1)):
            report = run_validation(tmp_path, skip_tests=True)

        assert isinstance(report, RCReport)
        assert report.version == "0.9.0"
        assert len(report.gates) == 8

    def test_gate_ids_are_sequential(self, tmp_path):
        (tmp_path / "version.py").write_text('__version__ = "1.0.0"\n')

        with mock.patch("rc_validation._run_command", return_value=(0, "", "", 0.0)):
            report = run_validation(tmp_path, skip_tests=True)

        gate_ids = [g.gate_id for g in report.gates]
        expected = [f"RC-{i:03d}" for i in range(1, 9)]
        assert gate_ids == expected


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------


class TestPrintReport:
    """Tests for print_report output."""

    def test_does_not_crash_empty(self, capsys):
        report = RCReport()
        print_report(report)
        captured = capsys.readouterr()
        assert "Release Candidate Validation" in captured.out

    def test_shows_version(self, capsys):
        report = RCReport(version="1.0.0")
        print_report(report)
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out

    def test_shows_gate_details(self, capsys):
        report = RCReport()
        report.add(GateResult("RC-001", "Version check", GateStatus.PASS, "Version: 1.0.0"))
        report.add(GateResult("RC-003", "Lint", GateStatus.FAIL, "3 errors", 0.5, "line1\nline2"))
        print_report(report)
        captured = capsys.readouterr()
        assert "RC-001" in captured.out
        assert "RC-003" in captured.out
        assert "line1" in captured.out
        assert "PASS" in captured.out or "pass" in captured.out.lower()


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """Tests for _find_project_root."""

    def test_returns_path(self):
        result = _find_project_root()
        assert isinstance(result, Path)
