"""Tests for CI/CD configuration files.

Validates that the security audit and SBOM generation workflows are properly
configured, and that the deliberately-disabled Dependabot posture (no version
updates, no auto-merge workflow) stays disabled — see
docs/release-candidate.md R4.
"""

from pathlib import Path

import pytest
import yaml

# Resolve repo root from this test file's location
REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
WORKFLOWS_DIR = GITHUB_DIR / "workflows"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CI workflow — security-audit job
# ---------------------------------------------------------------------------

class TestCISecurityAudit:
    """Validate the security-audit job in ci.yml."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = WORKFLOWS_DIR / "ci.yml"
        assert self.path.exists(), f"ci.yml not found at {self.path}"
        self.cfg = _load_yaml(self.path)

    def test_security_audit_job_exists(self):
        assert "security-audit" in self.cfg["jobs"]

    def test_security_audit_is_blocking(self):
        """pip-audit is a blocking gate; continue-on-error must not be set."""
        job = self.cfg["jobs"]["security-audit"]
        assert job.get("continue-on-error") is not True

    def test_security_audit_has_blocking_steps(self):
        """Blocking pip-audit steps exist without || true."""
        job = self.cfg["jobs"]["security-audit"]
        step_names = [s.get("name", "") for s in job["steps"]]
        assert any("blocking" in n.lower() for n in step_names)

    def test_security_audit_installs_pip_audit(self):
        job = self.cfg["jobs"]["security-audit"]
        steps_text = yaml.dump(job["steps"])
        assert "pip-audit" in steps_text

    def test_security_audit_runs_npm_audit(self):
        job = self.cfg["jobs"]["security-audit"]
        steps_text = yaml.dump(job["steps"])
        assert "npm audit" in steps_text

    def test_security_audit_uploads_artifact(self):
        job = self.cfg["jobs"]["security-audit"]
        upload_steps = [
            s for s in job["steps"]
            if s.get("uses", "").startswith("actions/upload-artifact")
        ]
        assert len(upload_steps) >= 1


# ---------------------------------------------------------------------------
# CI workflow — generate-sbom job
# ---------------------------------------------------------------------------

class TestCISBOMGeneration:
    """Validate the generate-sbom job in ci.yml."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = WORKFLOWS_DIR / "ci.yml"
        assert self.path.exists()
        self.cfg = _load_yaml(self.path)

    def test_generate_sbom_job_exists(self):
        assert "generate-sbom" in self.cfg["jobs"]

    def test_sbom_depends_on_security_audit(self):
        job = self.cfg["jobs"]["generate-sbom"]
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "security-audit" in needs

    def test_sbom_only_runs_on_main_push(self):
        job = self.cfg["jobs"]["generate-sbom"]
        condition = job.get("if", "")
        assert "refs/heads/main" in condition
        assert "push" in condition

    def test_sbom_installs_cyclonedx(self):
        job = self.cfg["jobs"]["generate-sbom"]
        steps_text = yaml.dump(job["steps"])
        assert "cyclonedx" in steps_text.lower()

    def test_sbom_uploads_artifact(self):
        job = self.cfg["jobs"]["generate-sbom"]
        upload_steps = [
            s for s in job["steps"]
            if s.get("uses", "").startswith("actions/upload-artifact")
        ]
        assert len(upload_steps) >= 1
        # Check retention is set
        artifact_step = upload_steps[0]
        assert artifact_step.get("with", {}).get("retention-days") is not None


# ---------------------------------------------------------------------------
# Dependabot disabled posture (version updates + auto-merge removed 2026-07-16)
# ---------------------------------------------------------------------------

class TestDependabotDisabled:
    """Lock in the disabled Dependabot posture.

    Dependabot version updates and the auto-merge workflow were deliberately
    removed; see docs/release-candidate.md R4. These
    negative assertions guard against either surface silently returning.
    """

    def test_dependabot_config_absent(self):
        # Version updates deliberately disabled; see
        # docs/release-candidate.md R4.
        path = GITHUB_DIR / "dependabot.yml"
        assert not path.exists(), (
            "dependabot.yml should not exist — Dependabot version updates are "
            "deliberately disabled (research R4)"
        )

    def test_dependabot_auto_merge_workflow_absent(self):
        # Orphaned auto-merge workflow removed 2026-07-16.
        path = WORKFLOWS_DIR / "dependabot-auto-merge.yml"
        assert not path.exists(), (
            "dependabot-auto-merge.yml should not exist — orphan removed 2026-07-16"
        )


# ---------------------------------------------------------------------------
# Cross-cutting: all expected workflow files exist
# ---------------------------------------------------------------------------

class TestWorkflowFilesExist:
    """Ensure all expected workflow files are present."""

    EXPECTED_WORKFLOWS = [
        "ci.yml",
        "container-scan.yml",
        "helm-deploy.yml",
        "playwright-pr.yml",
        "sdk-publish.yml",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_WORKFLOWS)
    def test_workflow_exists(self, filename):
        path = WORKFLOWS_DIR / filename
        assert path.exists(), f"Workflow file {filename} not found"
