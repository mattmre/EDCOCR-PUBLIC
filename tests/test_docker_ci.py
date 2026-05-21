"""Tests for the Docker publish CI workflow (.github/workflows/docker-publish.yml).

Validates that the workflow file exists, is valid YAML, and has the expected
structure for building and pushing Docker images to GHCR.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"


@pytest.fixture(scope="module")
def workflow():
    """Load and parse the docker-publish workflow YAML."""
    assert WORKFLOW_PATH.exists(), f"Workflow file not found: {WORKFLOW_PATH}"
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(content)


def _get_triggers(workflow):
    """Get the 'on' trigger section from the workflow.

    YAML's bare ``on`` is parsed by PyYAML as boolean True, so the dict
    key is ``True`` rather than the string ``"on"``.
    """
    return workflow.get(True) or workflow.get("on")


class TestWorkflowFileExists:
    """Verify the workflow file exists and is valid YAML."""

    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists()

    def test_workflow_is_valid_yaml(self, workflow):
        assert isinstance(workflow, dict)
        assert "name" in workflow
        assert workflow["name"] == "Docker Publish"


class TestWorkflowTriggers:
    """Verify the workflow triggers on the correct events."""

    def test_has_push_trigger(self, workflow):
        triggers = _get_triggers(workflow)
        assert triggers is not None, "Missing 'on' trigger section"
        assert "push" in triggers

    def test_push_triggers_on_main_branch(self, workflow):
        triggers = _get_triggers(workflow)
        branches = triggers["push"]["branches"]
        assert "main" in branches

    def test_push_triggers_on_version_tags(self, workflow):
        triggers = _get_triggers(workflow)
        tags = triggers["push"]["tags"]
        assert "v*" in tags

    def test_has_workflow_dispatch(self, workflow):
        triggers = _get_triggers(workflow)
        assert "workflow_dispatch" in triggers

    def test_no_pull_request_trigger(self, workflow):
        """Workflow should not trigger on PRs (only main/tags push)."""
        triggers = _get_triggers(workflow)
        assert "pull_request" not in triggers


class TestWorkflowPermissions:
    """Verify least-privilege permissions."""

    def test_has_permissions(self, workflow):
        assert "permissions" in workflow

    def test_contents_read(self, workflow):
        assert workflow["permissions"]["contents"] == "read"

    def test_packages_write(self, workflow):
        assert workflow["permissions"]["packages"] == "write"


class TestBuildJobs:
    """Verify all build jobs exist with correct structure."""

    EXPECTED_JOBS = [
        "build-main-image",
        "build-coordinator-image",
        "build-worker-image",
        "build-worker-ocr-image",
        "build-worker-nlp-image",
        "build-worker-layoutlm-image",
    ]

    @pytest.mark.parametrize("job_name", EXPECTED_JOBS)
    def test_job_exists(self, workflow, job_name):
        assert job_name in workflow["jobs"], f"Missing job: {job_name}"

    def test_has_main_image_job(self, workflow):
        assert "build-main-image" in workflow["jobs"]

    def test_has_coordinator_image_job(self, workflow):
        assert "build-coordinator-image" in workflow["jobs"]

    @pytest.mark.parametrize("job_name", EXPECTED_JOBS)
    def test_job_runs_on_ubuntu(self, workflow, job_name):
        assert workflow["jobs"][job_name]["runs-on"] == "ubuntu-latest"

    @pytest.mark.parametrize("job_name", EXPECTED_JOBS)
    def test_job_has_timeout(self, workflow, job_name):
        assert "timeout-minutes" in workflow["jobs"][job_name]

    def test_main_image_runs_on_ubuntu(self, workflow):
        assert workflow["jobs"]["build-main-image"]["runs-on"] == "ubuntu-latest"

    def test_coordinator_image_runs_on_ubuntu(self, workflow):
        assert workflow["jobs"]["build-coordinator-image"]["runs-on"] == "ubuntu-latest"

    def test_main_image_has_timeout(self, workflow):
        assert "timeout-minutes" in workflow["jobs"]["build-main-image"]

    def test_coordinator_image_has_timeout(self, workflow):
        assert "timeout-minutes" in workflow["jobs"]["build-coordinator-image"]


class TestGHCRLogin:
    """Verify GHCR login steps are configured correctly."""

    def _find_step(self, job, uses_prefix):
        """Find a step by its 'uses' field prefix."""
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    def test_main_image_has_ghcr_login(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        login_step = self._find_step(job, "docker/login-action")
        assert login_step is not None, "Missing docker/login-action step"
        assert login_step["with"]["registry"] == "ghcr.io"

    def test_coordinator_image_has_ghcr_login(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        login_step = self._find_step(job, "docker/login-action")
        assert login_step is not None, "Missing docker/login-action step"
        assert login_step["with"]["registry"] == "ghcr.io"

    def test_login_uses_github_actor(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        login_step = self._find_step(job, "docker/login-action")
        assert login_step["with"]["username"] == "${{ github.actor }}"

    def test_login_uses_github_token(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        login_step = self._find_step(job, "docker/login-action")
        assert login_step["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"


class TestDockerBuildPush:
    """Verify build-push steps use correct configuration."""

    def _find_step(self, job, uses_prefix):
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    def test_main_image_uses_build_push_v6(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step is not None
        assert step["uses"] == "docker/build-push-action@v7"

    def test_coordinator_uses_build_push_v6(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step is not None
        assert step["uses"] == "docker/build-push-action@v7"

    def test_main_image_dockerfile(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["file"] == "./Dockerfile"

    def test_coordinator_dockerfile(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["file"] == "./coordinator/Dockerfile.coordinator"

    def test_main_image_context(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["context"] == "."

    def test_coordinator_context(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["context"] == "./coordinator"

    def test_push_conditional_on_non_pr(self, workflow):
        """Push should only happen on non-PR events (main push, tags)."""
        for job_name in ("build-main-image", "build-coordinator-image"):
            job = workflow["jobs"][job_name]
            step = self._find_step(job, "docker/build-push-action")
            push_value = step["with"]["push"]
            assert "pull_request" in str(push_value), (
                f"{job_name}: push should be conditional on non-PR event"
            )


class TestBuildCaching:
    """Verify GitHub Actions build cache is configured."""

    def _find_step(self, job, uses_prefix):
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    def test_main_image_cache_from(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["cache-from"] == "type=gha"

    def test_main_image_cache_to(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["cache-to"] == "type=gha,mode=max"

    def test_coordinator_cache_from(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["cache-from"] == "type=gha"

    def test_coordinator_cache_to(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["cache-to"] == "type=gha,mode=max"


class TestMetadataAction:
    """Verify Docker metadata action is configured for proper tagging."""

    def _find_step(self, job, uses_prefix):
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    def test_main_image_metadata_action(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/metadata-action")
        assert step is not None
        assert "ghcr.io" in step["with"]["images"]
        assert "ocr-pipeline" in step["with"]["images"]

    def test_coordinator_metadata_action(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/metadata-action")
        assert step is not None
        assert "ghcr.io" in step["with"]["images"]
        assert "coordinator" in step["with"]["images"]

    def test_main_image_has_semver_tags(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/metadata-action")
        tags = step["with"]["tags"]
        assert "type=semver" in tags

    def test_main_image_has_sha_tag(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/metadata-action")
        tags = step["with"]["tags"]
        assert "type=sha" in tags

    def test_main_image_has_branch_tag(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/metadata-action")
        tags = step["with"]["tags"]
        assert "type=ref,event=branch" in tags


class TestWorkerBuildPush:
    """Verify worker build jobs have correct Dockerfile and image configuration."""

    WORKER_JOBS = {
        "build-worker-image": {
            "file": "./coordinator/Dockerfile.worker",
            "image_suffix": "ocr-worker",
        },
        "build-worker-ocr-image": {
            "file": "./coordinator/Dockerfile.worker.ocr",
            "image_suffix": "ocr-worker-ocr",
        },
        "build-worker-nlp-image": {
            "file": "./coordinator/Dockerfile.worker.nlp",
            "image_suffix": "ocr-worker-nlp",
        },
        "build-worker-layoutlm-image": {
            "file": "./coordinator/Dockerfile.worker.layoutlm",
            "image_suffix": "ocr-worker-layoutlm",
        },
    }

    def _find_step(self, job, uses_prefix):
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_dockerfile(self, workflow, job_name):
        expected = self.WORKER_JOBS[job_name]["file"]
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/build-push-action")
        assert step is not None, f"Missing build-push step in {job_name}"
        assert step["with"]["file"] == expected

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_context(self, workflow, job_name):
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/build-push-action")
        assert step["with"]["context"] == "./coordinator"

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_metadata_image(self, workflow, job_name):
        expected_suffix = self.WORKER_JOBS[job_name]["image_suffix"]
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/metadata-action")
        assert step is not None, f"Missing metadata step in {job_name}"
        assert expected_suffix in step["with"]["images"]

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_ghcr_login(self, workflow, job_name):
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/login-action")
        assert step is not None, f"Missing login step in {job_name}"
        assert step["with"]["registry"] == "ghcr.io"

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_push_conditional(self, workflow, job_name):
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/build-push-action")
        push_value = step["with"]["push"]
        assert "pull_request" in str(push_value)

    @pytest.mark.parametrize("job_name", list(WORKER_JOBS.keys()))
    def test_worker_skip_model_preload(self, workflow, job_name):
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/build-push-action")
        build_args = step["with"].get("build-args", "")
        assert "SKIP_MODEL_PRELOAD=1" in build_args


class TestBuildxSetup:
    """Verify Docker Buildx is configured for all jobs."""

    ALL_JOBS = [
        "build-main-image",
        "build-coordinator-image",
        "build-worker-image",
        "build-worker-ocr-image",
        "build-worker-nlp-image",
        "build-worker-layoutlm-image",
    ]

    def _find_step(self, job, uses_prefix):
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(uses_prefix):
                return step
        return None

    @pytest.mark.parametrize("job_name", ALL_JOBS)
    def test_job_has_buildx(self, workflow, job_name):
        job = workflow["jobs"][job_name]
        step = self._find_step(job, "docker/setup-buildx-action")
        assert step is not None, f"Missing buildx setup in {job_name}"

    def test_main_image_has_buildx(self, workflow):
        job = workflow["jobs"]["build-main-image"]
        step = self._find_step(job, "docker/setup-buildx-action")
        assert step is not None

    def test_coordinator_has_buildx(self, workflow):
        job = workflow["jobs"]["build-coordinator-image"]
        step = self._find_step(job, "docker/setup-buildx-action")
        assert step is not None
