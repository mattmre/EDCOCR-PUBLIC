"""Tests for the container-scan.yml GitHub Actions workflow.

Validates that the Trivy container image scanning workflow is
properly configured with correct triggers, matrix images,
SARIF upload, and failure behavior.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOWS_DIR / "container-scan.yml"


def _get_on_key(workflow: dict) -> dict:
    """Get the 'on' trigger config from a parsed GitHub Actions workflow.

    PyYAML parses the YAML key ``on:`` as boolean True, so we look up
    the trigger configuration under the boolean key first, then fall
    back to the string ``"on"`` for safety.
    """
    if True in workflow:
        return workflow[True]
    return workflow.get("on", {})


@pytest.fixture(scope="module")
def workflow():
    """Load and parse the container-scan workflow YAML."""
    assert WORKFLOW_PATH.exists(), (
        f"container-scan.yml not found at {WORKFLOW_PATH}"
    )
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Workflow-level structure
# ---------------------------------------------------------------------------


class TestWorkflowStructure:
    """Validate top-level workflow structure."""

    def test_yaml_is_parseable(self, workflow):
        assert isinstance(workflow, dict)

    def test_has_name(self, workflow):
        assert "name" in workflow
        assert "container" in workflow["name"].lower() or "scan" in workflow["name"].lower()

    def test_has_jobs(self, workflow):
        assert "jobs" in workflow
        assert len(workflow["jobs"]) >= 1

    def test_scan_images_job_exists(self, workflow):
        assert "scan-images" in workflow["jobs"]


# ---------------------------------------------------------------------------
# Trigger configuration
# ---------------------------------------------------------------------------


class TestTriggers:
    """Validate workflow trigger paths and events."""

    def test_triggers_on_push(self, workflow):
        triggers = _get_on_key(workflow)
        assert "push" in triggers

    def test_triggers_on_pull_request(self, workflow):
        triggers = _get_on_key(workflow)
        assert "pull_request" in triggers

    def test_triggers_on_schedule(self, workflow):
        triggers = _get_on_key(workflow)
        assert "schedule" in triggers
        schedules = triggers["schedule"]
        assert len(schedules) >= 1
        assert "cron" in schedules[0]

    def test_push_targets_main_branch(self, workflow):
        triggers = _get_on_key(workflow)
        push_branches = triggers["push"].get("branches", [])
        assert "main" in push_branches

    def test_push_has_dockerfile_path_filter(self, workflow):
        triggers = _get_on_key(workflow)
        push_paths = triggers["push"].get("paths", [])
        assert "Dockerfile" in push_paths

    def test_push_has_coordinator_dockerfile_path_filter(self, workflow):
        triggers = _get_on_key(workflow)
        push_paths = triggers["push"].get("paths", [])
        coordinator_paths = [p for p in push_paths if "coordinator/Dockerfile" in p]
        assert len(coordinator_paths) >= 1

    def test_push_has_requirements_path_filter(self, workflow):
        triggers = _get_on_key(workflow)
        push_paths = triggers["push"].get("paths", [])
        req_paths = [p for p in push_paths if "requirements.txt" in p]
        assert len(req_paths) >= 1

    def test_pr_has_same_path_filters_as_push(self, workflow):
        triggers = _get_on_key(workflow)
        push_paths = set(triggers["push"].get("paths", []))
        pr_paths = set(triggers["pull_request"].get("paths", []))
        assert push_paths == pr_paths


# ---------------------------------------------------------------------------
# Matrix configuration
# ---------------------------------------------------------------------------


class TestMatrix:
    """Validate the build matrix includes required images."""

    def _get_matrix_includes(self, workflow):
        job = workflow["jobs"]["scan-images"]
        return job["strategy"]["matrix"]["include"]

    def test_matrix_has_all_image_entries(self, workflow):
        includes = self._get_matrix_includes(workflow)
        assert len(includes) >= 6, (
            f"Expected at least 6 matrix entries (root + coordinator + 4 workers), "
            f"got {len(includes)}"
        )

    def test_matrix_includes_main_dockerfile(self, workflow):
        includes = self._get_matrix_includes(workflow)
        main_entries = [
            e for e in includes
            if e.get("dockerfile") == "Dockerfile"
        ]
        assert len(main_entries) == 1
        assert main_entries[0]["image"] == "ocr-local"

    def test_matrix_includes_coordinator_dockerfile(self, workflow):
        includes = self._get_matrix_includes(workflow)
        coord_entries = [
            e for e in includes
            if "coordinator" in e.get("dockerfile", "").lower()
        ]
        assert len(coord_entries) >= 1
        assert coord_entries[0]["image"] == "ocr-coordinator"

    @pytest.mark.parametrize("image_name", [
        "ocr-worker", "ocr-worker-ocr", "ocr-worker-nlp", "ocr-worker-layoutlm",
    ])
    def test_matrix_includes_worker_image(self, workflow, image_name):
        includes = self._get_matrix_includes(workflow)
        matching = [e for e in includes if e.get("image") == image_name]
        assert len(matching) == 1, (
            f"Expected exactly 1 matrix entry for image '{image_name}', "
            f"found {len(matching)}"
        )

    def test_matrix_entries_have_required_keys(self, workflow):
        includes = self._get_matrix_includes(workflow)
        for entry in includes:
            assert "dockerfile" in entry, f"Missing 'dockerfile' in matrix entry: {entry}"
            assert "context" in entry, f"Missing 'context' in matrix entry: {entry}"
            assert "image" in entry, f"Missing 'image' in matrix entry: {entry}"


# ---------------------------------------------------------------------------
# Trivy scanner steps
# ---------------------------------------------------------------------------


class TestTrivySteps:
    """Validate Trivy scan step configuration."""

    def _get_steps(self, workflow):
        return workflow["jobs"]["scan-images"]["steps"]

    def _find_trivy_steps(self, workflow):
        steps = self._get_steps(workflow)
        return [
            s for s in steps
            if "Trivy" in s.get("name", "") and "run" in s
        ]

    def test_has_trivy_scan_steps(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        assert len(trivy_steps) >= 2, "Expected both SARIF and table Trivy steps"

    def test_trivy_cli_image_is_pinned(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        for step in trivy_steps:
            run_cmd = step.get("run", "")
            assert "aquasec/trivy:0.69.3" in run_cmd, (
                "Trivy CLI image should be pinned to an explicit version"
            )

    def test_sarif_output_step_exists(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        sarif_steps = [
            s for s in trivy_steps
            if "(SARIF)" in s.get("name", "")
        ]
        assert len(sarif_steps) >= 1, "No SARIF format Trivy step found"

    def test_sarif_step_has_output_file(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        sarif_steps = [
            s for s in trivy_steps
            if "(SARIF)" in s.get("name", "")
        ]
        run_cmd = sarif_steps[0].get("run", "")
        assert "--format sarif" in run_cmd
        assert "--output" in run_cmd
        assert ".sarif" in run_cmd
        assert "cp " in run_cmd, "SARIF step must copy the generated file into the workspace"

    def test_exit_code_fails_on_critical_high(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        sarif_steps = [
            s for s in trivy_steps
            if "(SARIF)" in s.get("name", "")
        ]
        step = sarif_steps[0]
        run_cmd = step.get("run", "")
        assert "--exit-code 1" in run_cmd, (
            "Trivy SARIF step must fail on CRITICAL/HIGH findings"
        )
        assert "--severity CRITICAL,HIGH" in run_cmd

    def test_trivy_steps_scan_exported_tarballs(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        for step in trivy_steps:
            run_cmd = step.get("run", "")
            assert "--input" in run_cmd
            assert "runner.temp" in run_cmd
            assert "-scan.tar" in run_cmd

    def test_trivy_steps_set_timeout(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        for step in trivy_steps:
            run_cmd = step.get("run", "")
            assert "--timeout 15m" in run_cmd, (
                "Trivy CLI steps should set an explicit timeout for larger images"
            )

    def test_table_output_step_exists(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        table_steps = [
            s for s in trivy_steps
            if "(table output" in s.get("name", "")
        ]
        assert len(table_steps) >= 1, "No table format Trivy step found for PR visibility"

    def test_table_step_includes_medium_severity(self, workflow):
        trivy_steps = self._find_trivy_steps(workflow)
        table_steps = [
            s for s in trivy_steps
            if "(table output" in s.get("name", "")
        ]
        run_cmd = table_steps[0].get("run", "")
        assert "--format table" in run_cmd
        assert "--severity CRITICAL,HIGH,MEDIUM" in run_cmd, (
            "Table output should include MEDIUM severity"
        )


# ---------------------------------------------------------------------------
# SARIF upload step
# ---------------------------------------------------------------------------


class TestSarifUpload:
    """Validate SARIF results are uploaded to GitHub Security tab."""

    def _get_steps(self, workflow):
        return workflow["jobs"]["scan-images"]["steps"]

    def test_sarif_upload_step_exists(self, workflow):
        steps = self._get_steps(workflow)
        upload_steps = [
            s for s in steps
            if "codeql-action/upload-sarif" in s.get("uses", "")
        ]
        assert len(upload_steps) >= 1, "No codeql-action/upload-sarif step found"

    def test_sarif_upload_runs_always(self, workflow):
        steps = self._get_steps(workflow)
        upload_steps = [
            s for s in steps
            if "codeql-action/upload-sarif" in s.get("uses", "")
        ]
        step = upload_steps[0]
        assert step.get("if") == "always()", (
            "SARIF upload must run always() to capture results even on failure"
        )

    def test_sarif_upload_references_sarif_file(self, workflow):
        steps = self._get_steps(workflow)
        upload_steps = [
            s for s in steps
            if "codeql-action/upload-sarif" in s.get("uses", "")
        ]
        step = upload_steps[0]
        sarif_file = step.get("with", {}).get("sarif_file", "")
        assert sarif_file.endswith(".sarif"), (
            f"SARIF upload must reference a .sarif file, got: {sarif_file}"
        )


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------


class TestBuildStep:
    """Validate the Docker build step."""

    def _get_steps(self, workflow):
        return workflow["jobs"]["scan-images"]["steps"]

    def test_checkout_step_exists(self, workflow):
        steps = self._get_steps(workflow)
        checkout_steps = [
            s for s in steps
            if s.get("uses", "").startswith("actions/checkout")
        ]
        assert len(checkout_steps) >= 1

    def test_docker_build_step_exists(self, workflow):
        steps = self._get_steps(workflow)
        build_steps = [
            s for s in steps
            if "docker build" in s.get("run", "")
        ]
        assert len(build_steps) >= 1, "No docker build step found"

    def test_docker_build_uses_matrix_vars(self, workflow):
        steps = self._get_steps(workflow)
        build_steps = [
            s for s in steps
            if "docker build" in s.get("run", "")
        ]
        build_cmd = build_steps[0]["run"]
        assert "matrix.image" in build_cmd
        assert "matrix.dockerfile" in build_cmd
        assert "matrix.context" in build_cmd
