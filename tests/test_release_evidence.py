"""Tests for scripts/release_evidence.py -- release evidence bundle generator."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure scripts/ and project root are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scripts import release_evidence

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bundle_dir(tmp_path):
    """Provide a temporary evidence bundle directory."""
    d = tmp_path / "evidence-test"
    d.mkdir()
    return d


@pytest.fixture
def mock_subprocess():
    """Fixture that patches subprocess.run and returns the mock."""
    with mock.patch("release_evidence.subprocess.run") as m:
        yield m


# ---------------------------------------------------------------------------
# Helper to create a fake subprocess result
# ---------------------------------------------------------------------------


def _fake_result(stdout="", stderr="", returncode=0):
    """Build a fake subprocess.CompletedProcess."""
    return mock.Mock(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


# ---------------------------------------------------------------------------
# Test: _run_subprocess
# ---------------------------------------------------------------------------


class TestRunSubprocess:
    def test_success(self, mock_subprocess):
        mock_subprocess.return_value = _fake_result(stdout="hello", returncode=0)
        code, out, err, dur = release_evidence._run_subprocess(["echo", "hello"])
        assert code == 0
        assert out == "hello"
        assert dur >= 0

    def test_timeout(self, mock_subprocess):
        import subprocess as sp

        mock_subprocess.side_effect = sp.TimeoutExpired(cmd=["slow"], timeout=5)
        code, out, err, dur = release_evidence._run_subprocess(["slow"], timeout=5)
        assert code == -1
        assert "timed out" in err

    def test_file_not_found(self, mock_subprocess):
        mock_subprocess.side_effect = FileNotFoundError("no such file")
        code, out, err, dur = release_evidence._run_subprocess(["missing"])
        assert code == -2
        assert "not found" in err.lower()

    def test_os_error(self, mock_subprocess):
        mock_subprocess.side_effect = OSError("something broke")
        code, out, err, dur = release_evidence._run_subprocess(["broken"])
        assert code == -3
        assert "OS error" in err


# ---------------------------------------------------------------------------
# Test: _sha256_file
# ---------------------------------------------------------------------------


class TestSha256File:
    def test_correct_hash(self, tmp_path):
        f = tmp_path / "test.txt"
        content = b"hello world"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert release_evidence._sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert release_evidence._sha256_file(f) == expected


# ---------------------------------------------------------------------------
# Test: _read_version
# ---------------------------------------------------------------------------


class TestReadVersion:
    def test_reads_current_version(self):
        version = release_evidence._read_version()
        # Should match what's in version.py
        assert version != "unknown"
        parts = version.split(".")
        assert len(parts) == 3

    def test_missing_version_file(self, monkeypatch):
        monkeypatch.setattr(release_evidence, "PROJECT_ROOT", Path("/nonexistent"))
        assert release_evidence._read_version() == "unknown"


# ---------------------------------------------------------------------------
# Test: check_version_consistency
# ---------------------------------------------------------------------------


class TestCheckVersionConsistency:
    def test_pass(self, bundle_dir, mock_subprocess):
        data = {
            "version_consistency": {
                "passed": True,
                "canonical": "1.2.0",
                "total_sources": 14,
                "mismatches": {},
                "errors": [],
            }
        }
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(data), returncode=0
        )
        result = release_evidence.check_version_consistency(bundle_dir)
        assert result.status == release_evidence.PASS
        assert result.name == "version_consistency"
        assert result.category == "static"
        assert "14" in result.summary
        assert (bundle_dir / "version_consistency.json").exists()

    def test_fail_mismatch(self, bundle_dir, mock_subprocess):
        data = {
            "version_consistency": {
                "passed": False,
                "canonical": "1.2.0",
                "total_sources": 14,
                "mismatches": {"helm/Chart.yaml": "1.1.0"},
                "errors": [],
            }
        }
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(data), returncode=1
        )
        result = release_evidence.check_version_consistency(bundle_dir)
        assert result.status == release_evidence.FAIL
        assert "mismatch" in result.summary.lower()

    def test_subprocess_error(self, bundle_dir, mock_subprocess):
        import subprocess as sp

        mock_subprocess.side_effect = sp.TimeoutExpired(cmd=["x"], timeout=30)
        result = release_evidence.check_version_consistency(bundle_dir)
        assert result.status == release_evidence.ERROR
        assert "timed out" in result.summary.lower()

    def test_script_not_found(self, bundle_dir, monkeypatch):
        monkeypatch.setattr(release_evidence, "SCRIPT_DIR", Path("/nonexistent"))
        result = release_evidence.check_version_consistency(bundle_dir)
        assert result.status == release_evidence.ERROR
        assert "not found" in result.summary.lower()

    def test_invalid_json(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout="not json", returncode=0
        )
        result = release_evidence.check_version_consistency(bundle_dir)
        assert result.status == release_evidence.ERROR
        assert "parse" in result.summary.lower()


# ---------------------------------------------------------------------------
# Test: check_rc_validation
# ---------------------------------------------------------------------------


class TestCheckRcValidation:
    def _rc_data(self, overall="pass", gates=None):
        if gates is None:
            gates = [
                {"gate_id": "RC-001", "name": "Version check", "status": "pass", "message": "1.2.0"},
                {"gate_id": "RC-003", "name": "Lint", "status": "pass", "message": "clean"},
            ]
        return {
            "overall_status": overall,
            "gates": gates,
        }

    def test_pass(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._rc_data()), returncode=0
        )
        result = release_evidence.check_rc_validation(bundle_dir)
        assert result.status == release_evidence.PASS
        assert "2 passed" in result.summary
        assert (bundle_dir / "rc_validation.json").exists()

    def test_fail(self, bundle_dir, mock_subprocess):
        gates = [
            {"gate_id": "RC-001", "name": "Version", "status": "pass", "message": "ok"},
            {"gate_id": "RC-004", "name": "Tests", "status": "fail", "message": "3 failed"},
        ]
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._rc_data("fail", gates)), returncode=1
        )
        result = release_evidence.check_rc_validation(bundle_dir)
        assert result.status == release_evidence.FAIL
        assert "1 failed" in result.summary

    def test_skip_tests_flag(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._rc_data()), returncode=0
        )
        release_evidence.check_rc_validation(bundle_dir, skip_tests=True)
        call_args = mock_subprocess.call_args[0][0]
        assert "--skip-tests" in call_args

    def test_no_skip_tests_by_default(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._rc_data()), returncode=0
        )
        release_evidence.check_rc_validation(bundle_dir, skip_tests=False)
        call_args = mock_subprocess.call_args[0][0]
        assert "--skip-tests" not in call_args

    def test_script_not_found(self, bundle_dir, monkeypatch):
        monkeypatch.setattr(release_evidence, "SCRIPT_DIR", Path("/nonexistent"))
        result = release_evidence.check_rc_validation(bundle_dir)
        assert result.status == release_evidence.ERROR

    def test_warn_status(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._rc_data("warn")), returncode=0
        )
        result = release_evidence.check_rc_validation(bundle_dir)
        assert result.status == release_evidence.WARN


# ---------------------------------------------------------------------------
# Test: check_release_gate
# ---------------------------------------------------------------------------


class TestCheckReleaseGate:
    def _gate_data(self, overall="DRY-RUN-READY", gates=None, blockers=None):
        if gates is None:
            gates = [
                {"gate_name": "pull_request", "status": "passed", "detail": "ok"},
                {"gate_name": "workspace", "status": "passed", "detail": "clean"},
            ]
        return {
            "overall_status": overall,
            "gates": gates,
            "blockers": blockers or [],
        }

    def test_pass_dry_run(self, bundle_dir, mock_subprocess):
        data = self._gate_data()
        # subprocess writes json report to file
        def side_effect(cmd, **kwargs):
            # Find the --json-report path in the command
            for i, arg in enumerate(cmd):
                if arg == "--json-report" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_text(json.dumps(data), encoding="utf-8")
                    break
            return _fake_result(returncode=0)

        mock_subprocess.side_effect = side_effect
        result = release_evidence.check_release_gate(bundle_dir, pull_request=536)
        assert result.status == release_evidence.PASS
        assert "DRY-RUN-READY" in result.summary
        assert "536" in result.summary

    def test_blocked(self, bundle_dir, mock_subprocess):
        data = self._gate_data(
            "BLOCKED",
            blockers=["PR not merged"],
            gates=[
                {"gate_name": "pull_request", "status": "blocked", "detail": "PR #1 not merged"},
            ],
        )

        def side_effect(cmd, **kwargs):
            for i, arg in enumerate(cmd):
                if arg == "--json-report" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_text(json.dumps(data), encoding="utf-8")
                    break
            return _fake_result(returncode=1)

        mock_subprocess.side_effect = side_effect
        result = release_evidence.check_release_gate(bundle_dir)
        assert result.status == release_evidence.WARN
        assert "BLOCKED" in result.summary
        assert any("BLOCKER" in d for d in result.details)

    def test_script_not_found(self, bundle_dir, monkeypatch):
        monkeypatch.setattr(release_evidence, "SCRIPT_DIR", Path("/nonexistent"))
        result = release_evidence.check_release_gate(bundle_dir)
        assert result.status == release_evidence.ERROR

    def test_no_json_report_written(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout="Some text output", returncode=0
        )
        result = release_evidence.check_release_gate(bundle_dir)
        assert result.status == release_evidence.PASS
        assert "passed" in result.summary.lower()


# ---------------------------------------------------------------------------
# Test: check_cutover_preflight
# ---------------------------------------------------------------------------


class TestCheckCutoverPreflight:
    def _preflight_data(self, verdict="READY"):
        return {
            "verdict": verdict,
            "summary": {"pass": 5, "fail": 0, "warn": 1, "skip": 2, "total": 8},
            "checks": [
                {"name": "env_file", "status": "PASS", "details": "ok"},
                {"name": "connectivity", "status": "SKIP", "details": "skipped"},
            ],
        }

    def test_ready(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._preflight_data("READY")), returncode=0
        )
        result = release_evidence.check_cutover_preflight(bundle_dir)
        assert result.status == release_evidence.PASS
        assert result.category == "operational"
        assert "READY" in result.summary

    def test_not_ready(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._preflight_data("NOT READY")), returncode=1
        )
        result = release_evidence.check_cutover_preflight(bundle_dir)
        assert result.status == release_evidence.FAIL

    def test_partial(self, bundle_dir, mock_subprocess):
        mock_subprocess.return_value = _fake_result(
            stdout=json.dumps(self._preflight_data("PARTIAL")), returncode=2
        )
        result = release_evidence.check_cutover_preflight(bundle_dir)
        assert result.status == release_evidence.WARN

    def test_script_not_found(self, bundle_dir, monkeypatch):
        monkeypatch.setattr(release_evidence, "SCRIPT_DIR", Path("/nonexistent"))
        result = release_evidence.check_cutover_preflight(bundle_dir)
        assert result.status == release_evidence.ERROR


# ---------------------------------------------------------------------------
# Test: compute_verdict
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    def _check(self, status):
        return release_evidence.EvidenceCheck(
            name="test", category="static", status=status, summary="x"
        )

    def test_all_pass(self):
        checks = [self._check("PASS"), self._check("PASS")]
        assert release_evidence.compute_verdict(checks) == "PASS"

    def test_any_fail(self):
        checks = [self._check("PASS"), self._check("FAIL")]
        assert release_evidence.compute_verdict(checks) == "FAIL"

    def test_any_error(self):
        checks = [self._check("PASS"), self._check("ERROR")]
        assert release_evidence.compute_verdict(checks) == "FAIL"

    def test_warn_only(self):
        checks = [self._check("PASS"), self._check("WARN")]
        assert release_evidence.compute_verdict(checks) == "PARTIAL"

    def test_skip_is_pass(self):
        checks = [self._check("PASS"), self._check("SKIP")]
        assert release_evidence.compute_verdict(checks) == "PASS"

    def test_fail_overrides_warn(self):
        checks = [self._check("WARN"), self._check("FAIL")]
        assert release_evidence.compute_verdict(checks) == "FAIL"


# ---------------------------------------------------------------------------
# Test: verdict_exit_code
# ---------------------------------------------------------------------------


class TestVerdictExitCode:
    def test_pass_is_zero(self):
        assert release_evidence.verdict_exit_code("PASS") == 0

    def test_fail_is_one(self):
        assert release_evidence.verdict_exit_code("FAIL") == 1

    def test_partial_is_two(self):
        assert release_evidence.verdict_exit_code("PARTIAL") == 2

    def test_unknown_is_two(self):
        assert release_evidence.verdict_exit_code("UNKNOWN") == 2


# ---------------------------------------------------------------------------
# Test: build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_manifest_structure(self, bundle_dir):
        art = bundle_dir / "test.json"
        art.write_text("{}", encoding="utf-8")

        checks = [
            release_evidence.EvidenceCheck(
                name="c1",
                category="static",
                status="PASS",
                summary="ok",
                artifact_path="test.json",
            ),
            release_evidence.EvidenceCheck(
                name="c2",
                category="static",
                status="PASS",
                summary="fine",
                artifact_path=None,
            ),
        ]
        manifest = release_evidence.build_manifest(checks, bundle_dir, "1.2.0", "PASS")

        assert manifest["version"] == "1.2.0"
        assert manifest["overall_status"] == "PASS"
        assert "timestamp" in manifest
        assert len(manifest["checks"]) == 2
        assert len(manifest["artifacts"]) == 1
        assert manifest["artifacts"][0]["name"] == "test.json"
        assert len(manifest["artifacts"][0]["sha256"]) == 64

    def test_manifest_artifact_sha256_correct(self, bundle_dir):
        art = bundle_dir / "data.json"
        content = b'{"hello": "world"}'
        art.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()

        checks = [
            release_evidence.EvidenceCheck(
                name="c1",
                category="static",
                status="PASS",
                summary="ok",
                artifact_path="data.json",
            ),
        ]
        manifest = release_evidence.build_manifest(checks, bundle_dir, "1.0.0", "PASS")
        assert manifest["artifacts"][0]["sha256"] == expected


# ---------------------------------------------------------------------------
# Test: build_summary_md
# ---------------------------------------------------------------------------


class TestBuildSummaryMd:
    def test_contains_check_table(self):
        checks = [
            release_evidence.EvidenceCheck(
                name="version_check",
                category="static",
                status="PASS",
                summary="All versions match",
                duration_seconds=1.5,
            ),
        ]
        md = release_evidence.build_summary_md(checks, "PASS", "1.2.0")
        assert "# Release Evidence Summary" in md
        assert "version_check" in md
        assert "1.2.0" in md
        assert "PASS" in md
        assert "1.5s" in md

    def test_details_section(self):
        checks = [
            release_evidence.EvidenceCheck(
                name="test_check",
                category="static",
                status="FAIL",
                summary="failed",
                details=["detail line 1", "detail line 2"],
                artifact_path="report.json",
            ),
        ]
        md = release_evidence.build_summary_md(checks, "FAIL", "1.2.0")
        assert "### test_check" in md
        assert "detail line 1" in md
        assert "report.json" in md


# ---------------------------------------------------------------------------
# Test: build_text_summary
# ---------------------------------------------------------------------------


class TestBuildTextSummary:
    def test_basic_output(self):
        checks = [
            release_evidence.EvidenceCheck(
                name="check_a",
                category="static",
                status="PASS",
                summary="all good",
                duration_seconds=2.0,
            ),
        ]
        text = release_evidence.build_text_summary(checks, "PASS")
        assert "[PASS]" in text
        assert "check_a" in text
        assert "Overall: PASS" in text

    def test_fail_details_shown(self):
        checks = [
            release_evidence.EvidenceCheck(
                name="check_b",
                category="static",
                status="FAIL",
                summary="something broke",
                details=["line 1", "line 2"],
            ),
        ]
        text = release_evidence.build_text_summary(checks, "FAIL")
        assert "[FAIL]" in text
        assert "line 1" in text


# ---------------------------------------------------------------------------
# Test: run_evidence (end-to-end with mocked subprocesses)
# ---------------------------------------------------------------------------


class TestRunEvidence:
    def _mock_all_checks(self, monkeypatch, statuses=None):
        """Replace all check functions with mocks returning given statuses."""
        if statuses is None:
            statuses = {
                "version_consistency": "PASS",
                "rc_validation": "PASS",
                "release_gate": "PASS",
                "cutover_preflight": "PASS",
            }

        def mock_vc(bundle_dir):
            return release_evidence.EvidenceCheck(
                name="version_consistency",
                category="static",
                status=statuses.get("version_consistency", "PASS"),
                summary="mocked",
            )

        def mock_rc(bundle_dir, *, skip_tests=False):
            return release_evidence.EvidenceCheck(
                name="rc_validation",
                category="static",
                status=statuses.get("rc_validation", "PASS"),
                summary="mocked",
            )

        def mock_rg(bundle_dir, *, pull_request=None):
            return release_evidence.EvidenceCheck(
                name="release_gate",
                category="static",
                status=statuses.get("release_gate", "PASS"),
                summary="mocked",
            )

        def mock_cp(bundle_dir):
            return release_evidence.EvidenceCheck(
                name="cutover_preflight",
                category="operational",
                status=statuses.get("cutover_preflight", "PASS"),
                summary="mocked",
            )

        monkeypatch.setattr(release_evidence, "check_version_consistency", mock_vc)
        monkeypatch.setattr(release_evidence, "check_rc_validation", mock_rc)
        monkeypatch.setattr(release_evidence, "check_release_gate", mock_rg)
        monkeypatch.setattr(release_evidence, "check_cutover_preflight", mock_cp)

    def test_tier1_only_by_default(self, tmp_path, monkeypatch):
        self._mock_all_checks(monkeypatch)
        exit_code = release_evidence.run_evidence(evidence_dir=tmp_path / "ev")
        assert exit_code == 0
        manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
        # Only 3 checks (Tier 1)
        assert len(manifest["checks"]) == 3
        assert manifest["overall_status"] == "PASS"

    def test_full_includes_tier2(self, tmp_path, monkeypatch):
        self._mock_all_checks(monkeypatch)
        exit_code = release_evidence.run_evidence(
            full=True, evidence_dir=tmp_path / "ev"
        )
        assert exit_code == 0
        manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
        # 4 checks (Tier 1 + Tier 2)
        assert len(manifest["checks"]) == 4
        names = [c["name"] for c in manifest["checks"]]
        assert "cutover_preflight" in names

    def test_fail_exit_code(self, tmp_path, monkeypatch):
        self._mock_all_checks(
            monkeypatch, {"version_consistency": "FAIL", "rc_validation": "PASS", "release_gate": "PASS"}
        )
        exit_code = release_evidence.run_evidence(evidence_dir=tmp_path / "ev")
        assert exit_code == 1

    def test_partial_exit_code(self, tmp_path, monkeypatch):
        self._mock_all_checks(
            monkeypatch, {"version_consistency": "PASS", "rc_validation": "WARN", "release_gate": "PASS"}
        )
        exit_code = release_evidence.run_evidence(evidence_dir=tmp_path / "ev")
        assert exit_code == 2

    def test_bundle_files_created(self, tmp_path, monkeypatch):
        self._mock_all_checks(monkeypatch)
        ev_dir = tmp_path / "ev"
        release_evidence.run_evidence(evidence_dir=ev_dir)
        assert (ev_dir / "manifest.json").exists()
        assert (ev_dir / "summary.md").exists()

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        self._mock_all_checks(monkeypatch)
        release_evidence.run_evidence(evidence_dir=tmp_path / "ev", json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "overall_status" in data
        assert "checks" in data

    def test_skip_tests_propagated(self, tmp_path, monkeypatch):
        called_with_skip = {}

        def mock_rc(bundle_dir, *, skip_tests=False):
            called_with_skip["value"] = skip_tests
            return release_evidence.EvidenceCheck(
                name="rc_validation", category="static", status="PASS", summary="ok"
            )

        monkeypatch.setattr(release_evidence, "check_version_consistency",
                            lambda bd: release_evidence.EvidenceCheck(
                                name="vc", category="static", status="PASS", summary="ok"))
        monkeypatch.setattr(release_evidence, "check_rc_validation", mock_rc)
        monkeypatch.setattr(release_evidence, "check_release_gate",
                            lambda bd, pull_request=None: release_evidence.EvidenceCheck(
                                name="rg", category="static", status="PASS", summary="ok"))

        release_evidence.run_evidence(evidence_dir=tmp_path / "ev", skip_tests=True)
        assert called_with_skip["value"] is True

    def test_pull_request_propagated(self, tmp_path, monkeypatch):
        called_with_pr = {}

        def mock_rg(bundle_dir, *, pull_request=None):
            called_with_pr["value"] = pull_request
            return release_evidence.EvidenceCheck(
                name="rg", category="static", status="PASS", summary="ok"
            )

        monkeypatch.setattr(release_evidence, "check_version_consistency",
                            lambda bd: release_evidence.EvidenceCheck(
                                name="vc", category="static", status="PASS", summary="ok"))
        monkeypatch.setattr(release_evidence, "check_rc_validation",
                            lambda bd, skip_tests=False: release_evidence.EvidenceCheck(
                                name="rc", category="static", status="PASS", summary="ok"))
        monkeypatch.setattr(release_evidence, "check_release_gate", mock_rg)

        release_evidence.run_evidence(
            evidence_dir=tmp_path / "ev", pull_request=536
        )
        assert called_with_pr["value"] == 536


# ---------------------------------------------------------------------------
# Test: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = release_evidence.parse_args([])
        assert args.full is False
        assert args.skip_tests is False
        assert args.pull_request is None
        assert args.evidence_dir is None
        assert args.json_output is False

    def test_full_flag(self):
        args = release_evidence.parse_args(["--full"])
        assert args.full is True

    def test_skip_tests(self):
        args = release_evidence.parse_args(["--skip-tests"])
        assert args.skip_tests is True

    def test_pull_request(self):
        args = release_evidence.parse_args(["--pull-request", "536"])
        assert args.pull_request == 536

    def test_evidence_dir(self):
        args = release_evidence.parse_args(["--evidence-dir", "/tmp/test"])
        assert args.evidence_dir == Path("/tmp/test")

    def test_json_flag(self):
        args = release_evidence.parse_args(["--json"])
        assert args.json_output is True


# ---------------------------------------------------------------------------
# Test: _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_writes_file(self, bundle_dir):
        path = release_evidence._write_artifact(bundle_dir, "test.txt", "content")
        assert path.exists()
        assert path.read_text() == "content"
        assert path == bundle_dir / "test.txt"
