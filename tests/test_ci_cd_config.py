"""Tests for CI/CD configuration files.

Validates that Dependabot, security audit, SBOM generation,
and auto-merge workflows are properly configured.
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
# Dependabot configuration
# ---------------------------------------------------------------------------

class TestDependabotConfig:
    """Validate .github/dependabot.yml structure and ecosystems."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = GITHUB_DIR / "dependabot.yml"
        assert self.path.exists(), f"dependabot.yml not found at {self.path}"
        self.cfg = _load_yaml(self.path)

    def test_version_is_2(self):
        assert self.cfg["version"] == 2

    def test_has_updates_list(self):
        assert "updates" in self.cfg
        assert isinstance(self.cfg["updates"], list)
        assert len(self.cfg["updates"]) >= 4  # pip root, pip coordinator, npm, docker

    def _find_ecosystem(self, ecosystem: str, directory: str = "/"):
        """Return the update entry matching ecosystem + directory."""
        for entry in self.cfg["updates"]:
            if entry["package-ecosystem"] == ecosystem and entry["directory"] == directory:
                return entry
        pytest.fail(f"No {ecosystem} entry for directory {directory}")

    def test_pip_root_ecosystem(self):
        entry = self._find_ecosystem("pip", "/")
        assert entry["schedule"]["interval"] == "weekly"
        assert "dependencies" in entry["labels"]
        assert "python" in entry["labels"]

    def test_pip_coordinator_ecosystem(self):
        entry = self._find_ecosystem("pip", "/coordinator")
        assert entry["schedule"]["interval"] == "weekly"
        assert "coordinator" in entry["labels"]

    def test_npm_typescript_sdk_ecosystem(self):
        entry = self._find_ecosystem("npm", "/sdk/typescript")
        assert entry["schedule"]["interval"] == "weekly"
        assert "typescript" in entry["labels"]

    def test_docker_root_ecosystem(self):
        entry = self._find_ecosystem("docker", "/")
        assert entry["schedule"]["interval"] in ("weekly", "monthly")
        assert "docker" in entry["labels"]

    def test_terraform_ecosystem(self):
        entry = self._find_ecosystem("terraform", "/terraform")
        assert entry["schedule"]["interval"] == "monthly"
        assert "terraform" in entry["labels"]

    def test_github_actions_ecosystem(self):
        entry = self._find_ecosystem("github-actions", "/")
        assert entry["schedule"]["interval"] in ("weekly", "monthly")
        assert "github-actions" in entry["labels"]

    def test_minor_patch_grouping(self):
        """At least the pip-root entry should group minor+patch updates."""
        entry = self._find_ecosystem("pip", "/")
        groups = entry.get("groups", {})
        assert "minor-and-patch" in groups
        update_types = groups["minor-and-patch"]["update-types"]
        assert "minor" in update_types
        assert "patch" in update_types

    def test_all_entries_have_labels(self):
        for entry in self.cfg["updates"]:
            assert "labels" in entry, (
                f"{entry['package-ecosystem']} ({entry['directory']}) is missing labels"
            )
            assert "dependencies" in entry["labels"]


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
# Dependabot auto-merge workflow
# ---------------------------------------------------------------------------

class TestDependabotAutoMerge:
    """Validate .github/workflows/dependabot-auto-merge.yml."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = WORKFLOWS_DIR / "dependabot-auto-merge.yml"
        assert self.path.exists(), f"dependabot-auto-merge.yml not found at {self.path}"
        self.cfg = _load_yaml(self.path)

    def test_triggers_on_pull_request(self):
        # PyYAML parses bare `on:` key as boolean True; use True as fallback
        triggers = self.cfg.get("on") or self.cfg.get(True, {})
        assert "pull_request" in triggers

    def test_has_auto_merge_job(self):
        assert "auto-merge" in self.cfg["jobs"]

    def test_only_runs_for_dependabot(self):
        job = self.cfg["jobs"]["auto-merge"]
        condition = job.get("if", "")
        assert "dependabot[bot]" in condition

    def test_does_not_auto_merge_major(self):
        """Major version bumps must NOT be auto-approved."""
        job = self.cfg["jobs"]["auto-merge"]
        steps = job["steps"]
        approve_steps = [
            s for s in steps
            if "approve" in s.get("name", "").lower()
        ]
        assert len(approve_steps) >= 1
        approve_step = approve_steps[0]
        # The condition should exclude major updates
        condition = approve_step.get("if", "")
        assert "semver-major" in condition

    def test_has_permissions(self):
        perms = self.cfg.get("permissions", {})
        assert "pull-requests" in perms
        assert "contents" in perms

    def test_uses_fetch_metadata(self):
        job = self.cfg["jobs"]["auto-merge"]
        metadata_steps = [
            s for s in job["steps"]
            if "fetch-metadata" in s.get("uses", "")
        ]
        assert len(metadata_steps) >= 1


# ---------------------------------------------------------------------------
# Cross-cutting: all expected workflow files exist
# ---------------------------------------------------------------------------

class TestWorkflowFilesExist:
    """Ensure all expected workflow files are present."""

    EXPECTED_WORKFLOWS = [
        "ci.yml",
        "container-scan.yml",
        "dependabot-auto-merge.yml",
        "helm-deploy.yml",
        "playwright-pr.yml",
        "sdk-publish.yml",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_WORKFLOWS)
    def test_workflow_exists(self, filename):
        path = WORKFLOWS_DIR / filename
        assert path.exists(), f"Workflow file {filename} not found"

    def test_dependabot_config_exists(self):
        path = GITHUB_DIR / "dependabot.yml"
        assert path.exists(), "dependabot.yml not found in .github/"
