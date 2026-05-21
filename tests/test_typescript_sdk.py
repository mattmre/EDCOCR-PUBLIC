"""Structural validation tests for the TypeScript SDK npm package.

These tests verify that the npm package structure is correct, required
files exist, and package metadata is valid. Since we cannot run TypeScript
tests from pytest, these tests focus on static validation of the package
layout and configuration.
"""

import json
import re
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).resolve().parent.parent / "sdk" / "typescript"


# ---------------------------------------------------------------------------
# package.json
# ---------------------------------------------------------------------------


class TestPackageJson:
    """Validate package.json structure and content."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = SDK_ROOT / "package.json"
        assert path.exists(), "package.json must exist"
        with open(path, encoding="utf-8") as f:
            self.pkg = json.load(f)

    def test_name(self):
        assert self.pkg["name"] == "@edcocr/sdk"

    def test_version_is_semver(self):
        version = self.pkg["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid semver: {version}"

    def test_version_matches_sdk_constant(self):
        """SDK_VERSION in src/client.ts should match package.json version."""
        client_ts = SDK_ROOT / "src" / "client.ts"
        content = client_ts.read_text(encoding="utf-8")
        match = re.search(r"SDK_VERSION\s*=\s*['\"](\d+\.\d+\.\d+)['\"]", content)
        assert match, "SDK_VERSION not found in client.ts"
        assert match.group(1) == self.pkg["version"]

    def test_main_entry(self):
        assert self.pkg["main"] == "dist/index.js"

    def test_types_entry(self):
        assert self.pkg["types"] == "dist/index.d.ts"

    def test_description(self):
        desc = self.pkg.get("description", "")
        assert len(desc) > 10, "Description should be meaningful"
        assert "OCR" in desc.upper()

    def test_license(self):
        assert self.pkg["license"] == "MIT"

    def test_engines_node_18_plus(self):
        engines = self.pkg.get("engines", {})
        assert "node" in engines
        assert "18" in engines["node"], "Should require Node.js 18+"

    def test_no_runtime_dependencies(self):
        """SDK should have zero runtime dependencies (fetch-based)."""
        deps = self.pkg.get("dependencies", {})
        assert len(deps) == 0, f"Expected zero dependencies, got: {list(deps.keys())}"

    def test_has_dev_dependencies(self):
        dev_deps = self.pkg.get("devDependencies", {})
        assert "typescript" in dev_deps
        assert "vitest" in dev_deps

    def test_has_build_script(self):
        scripts = self.pkg.get("scripts", {})
        assert "build" in scripts

    def test_has_test_script(self):
        scripts = self.pkg.get("scripts", {})
        assert "test" in scripts

    def test_has_repository(self):
        repo = self.pkg.get("repository", {})
        assert "url" in repo
        assert "EDCOCR" in repo["url"]

    def test_exports_field(self):
        exports = self.pkg.get("exports", {})
        assert "." in exports
        root = exports["."]
        assert "types" in root
        assert "import" in root or "require" in root

    def test_sideeffects_false(self):
        assert self.pkg.get("sideEffects") is False


# ---------------------------------------------------------------------------
# tsconfig.json
# ---------------------------------------------------------------------------


class TestTsconfigJson:
    """Validate tsconfig.json structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = SDK_ROOT / "tsconfig.json"
        assert path.exists(), "tsconfig.json must exist"
        with open(path, encoding="utf-8") as f:
            self.config = json.load(f)

    def test_strict_mode(self):
        opts = self.config.get("compilerOptions", {})
        assert opts.get("strict") is True

    def test_declaration_enabled(self):
        opts = self.config.get("compilerOptions", {})
        assert opts.get("declaration") is True

    def test_target_es2020_or_later(self):
        opts = self.config.get("compilerOptions", {})
        target = opts.get("target", "")
        assert target in ("ES2020", "ES2021", "ES2022", "ESNext")

    def test_outdir_is_dist(self):
        opts = self.config.get("compilerOptions", {})
        assert opts.get("outDir") == "dist"

    def test_rootdir_is_src(self):
        opts = self.config.get("compilerOptions", {})
        assert opts.get("rootDir") == "src"

    def test_includes_src(self):
        includes = self.config.get("include", [])
        assert any("src" in inc for inc in includes)


# ---------------------------------------------------------------------------
# Source file existence and content
# ---------------------------------------------------------------------------


class TestSourceFiles:
    """Validate that required source files exist and have expected content."""

    def test_index_ts_exists(self):
        assert (SDK_ROOT / "src" / "index.ts").exists()

    def test_client_ts_exists(self):
        assert (SDK_ROOT / "src" / "client.ts").exists()

    def test_models_ts_exists(self):
        assert (SDK_ROOT / "src" / "models.ts").exists()

    def test_errors_ts_exists(self):
        assert (SDK_ROOT / "src" / "errors.ts").exists()

    def test_index_exports_client(self):
        content = (SDK_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
        assert "EDCOCRClient" in content

    def test_index_exports_errors(self):
        content = (SDK_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
        for error_class in [
            "OCRLocalError",
            "AuthenticationError",
            "NotFoundError",
            "RateLimitError",
            "ServerError",
            "TimeoutError",
        ]:
            assert error_class in content, f"Missing export: {error_class}"

    def test_index_exports_models(self):
        content = (SDK_ROOT / "src" / "index.ts").read_text(encoding="utf-8")
        for model in ["JobStatus", "ClientConfig", "SubmitOptions"]:
            assert model in content, f"Missing export: {model}"

    def test_client_has_expected_methods(self):
        content = (SDK_ROOT / "src" / "client.ts").read_text(encoding="utf-8")
        expected_methods = [
            "healthCheck",
            "submitJob",
            "getStatus",
            "getResult",
            "cancelJob",
            "retryJob",
            "listJobs",
            "waitForCompletion",
            "submitAndWait",
            "downloadArtifact",
            "streamProgress",
            "close",
        ]
        for method in expected_methods:
            assert method in content, f"Missing method: {method}"

    def test_models_has_expected_interfaces(self):
        content = (SDK_ROOT / "src" / "models.ts").read_text(encoding="utf-8")
        expected = [
            "JobStatus",
            "JobProgress",
            "JobSubmitResponse",
            "JobStatusResponse",
            "JobListResponse",
            "JobResultResponse",
            "HealthResponse",
            "ClientConfig",
            "SubmitOptions",
            "ListJobsOptions",
            "WaitOptions",
            "WebSocketMessage",
        ]
        for name in expected:
            assert name in content, f"Missing type/interface: {name}"

    def test_errors_has_expected_classes(self):
        content = (SDK_ROOT / "src" / "errors.ts").read_text(encoding="utf-8")
        expected = [
            "OCRLocalError",
            "AuthenticationError",
            "NotFoundError",
            "RateLimitError",
            "ServerError",
            "TimeoutError",
            "ClientClosedError",
            "ConflictError",
        ]
        for name in expected:
            assert name in content, f"Missing error class: {name}"

    def test_errors_extend_base(self):
        """All custom errors should extend OCRLocalError."""
        content = (SDK_ROOT / "src" / "errors.ts").read_text(encoding="utf-8")
        subclasses = [
            "AuthenticationError",
            "NotFoundError",
            "RateLimitError",
            "ServerError",
            "TimeoutError",
            "ClientClosedError",
            "ConflictError",
        ]
        for cls in subclasses:
            assert re.search(
                rf"class\s+{cls}\s+extends\s+OCRLocalError", content
            ), f"{cls} should extend OCRLocalError"


# ---------------------------------------------------------------------------
# Client implementation details
# ---------------------------------------------------------------------------


class TestClientImplementation:
    """Verify key implementation patterns in the client source."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = (SDK_ROOT / "src" / "client.ts").read_text(encoding="utf-8")

    def test_uses_native_fetch(self):
        assert "fetch(" in self.content or "_fetch(" in self.content

    def test_abort_controller_for_timeout(self):
        assert "AbortController" in self.content

    def test_form_data_for_upload(self):
        assert "FormData" in self.content

    def test_user_agent_header(self):
        assert "ocr-local-typescript-sdk" in self.content

    def test_x_api_key_header(self):
        assert "X-API-Key" in self.content

    def test_retry_loop(self):
        assert re.search(r"for\s*\(\s*let\s+attempt", self.content)

    def test_exponential_backoff(self):
        assert re.search(r"2\s*\*\*\s*attempt", self.content)

    def test_sleep_between_retries(self):
        assert re.search(r"await\s+sleep\(", self.content)

    def test_health_endpoint_path(self):
        assert "/api/v1/health" in self.content

    def test_jobs_endpoint_path(self):
        assert "/api/v1/jobs" in self.content

    def test_websocket_endpoint_path(self):
        assert "/ws/jobs/" in self.content

    def test_encodes_job_id(self):
        """Job ID should be URI-encoded to prevent injection."""
        assert "encodeURIComponent" in self.content

    def test_handles_204_no_content(self):
        assert "204" in self.content

    def test_rate_limit_error_handling(self):
        assert "429" in self.content

    def test_conflict_error_handling(self):
        assert "409" in self.content


# ---------------------------------------------------------------------------
# Models alignment with API
# ---------------------------------------------------------------------------


class TestModelsApiAlignment:
    """Verify models match the server-side Pydantic schemas."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = (SDK_ROOT / "src" / "models.ts").read_text(encoding="utf-8")

    def test_job_status_enum_values(self):
        for value in ["queued", "processing", "completed", "failed", "cancelled"]:
            assert value in self.content, f"Missing JobStatus value: {value}"

    def test_priority_type(self):
        assert "urgent" in self.content
        assert "normal" in self.content
        assert "low" in self.content

    def test_docintel_mode_type(self):
        assert "layout_only" in self.content
        assert "tables_only" in self.content
        assert "full" in self.content

    def test_websocket_message_types(self):
        for msg_type in ["connected", "progress", "completed", "failed", "cancelled", "error", "pong"]:
            assert msg_type in self.content, f"Missing WebSocket message type: {msg_type}"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


class TestReadme:
    """Validate README.md exists and has essential content."""

    def test_readme_exists(self):
        assert (SDK_ROOT / "README.md").exists()

    def test_readme_has_install_instructions(self):
        content = (SDK_ROOT / "README.md").read_text(encoding="utf-8")
        assert "npm install" in content

    def test_readme_has_usage_example(self):
        content = (SDK_ROOT / "README.md").read_text(encoding="utf-8")
        assert "EDCOCRClient" in content

    def test_readme_has_error_handling(self):
        content = (SDK_ROOT / "README.md").read_text(encoding="utf-8")
        assert "AuthenticationError" in content

    def test_readme_has_browser_usage(self):
        content = (SDK_ROOT / "README.md").read_text(encoding="utf-8")
        assert "browser" in content.lower() or "Browser" in content


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class TestTestFiles:
    """Validate that TypeScript tests exist."""

    def test_client_test_exists(self):
        assert (SDK_ROOT / "tests" / "client.test.ts").exists()

    def test_vitest_config_exists(self):
        assert (SDK_ROOT / "vitest.config.ts").exists()


# ---------------------------------------------------------------------------
# CI workflow
# ---------------------------------------------------------------------------


class TestCIWorkflow:
    """Validate CI workflow for SDK publishing."""

    def test_sdk_publish_workflow_exists(self):
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "sdk-publish.yml"
        )
        assert workflow.exists(), "sdk-publish.yml workflow must exist"

    def test_sdk_publish_triggers_on_tag(self):
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "sdk-publish.yml"
        )
        content = workflow.read_text(encoding="utf-8")
        assert "sdk-ts-v" in content, "Should trigger on sdk-ts-v* tags"

    def test_sdk_publish_has_npm_step(self):
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "sdk-publish.yml"
        )
        content = workflow.read_text(encoding="utf-8")
        assert "npm" in content.lower()


# ---------------------------------------------------------------------------
# Legacy file compatibility
# ---------------------------------------------------------------------------


class TestLegacyCompatibility:
    """Verify the original ocr_client.ts is still present for backward compat."""

    def test_original_file_still_exists(self):
        """The original single-file SDK should remain alongside the package."""
        assert (SDK_ROOT / "ocr_client.ts").exists()
