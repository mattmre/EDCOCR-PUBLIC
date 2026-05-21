"""Unit tests for scripts/validate_deployment.py.

Covers CLI argument parsing, health check logic, schema validation,
report generation, topology profile selection, and version check logic.

Run with: python -m pytest tests/test_deployment_validation.py -v
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from scripts.validate_deployment import (
    ALL_PROFILES,
    DEFAULT_TIMEOUT_SECONDS,
    EXPECTED_SCHEMA_TYPES,
    PROFILE_KUBERNETES,
    PROFILE_MULTI_GPU,
    PROFILE_SINGLE_NODE,
    CheckResult,
    DeploymentReport,
    build_parser,
    check_detailed_health,
    check_health,
    check_job_submission,
    check_schema_list,
    check_schema_retrieval,
    check_version_header,
    format_json_report,
    format_markdown_report,
    main,
    run_validation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_http_get_mock(responses: dict):
    """Create a side_effect function for _http_get that returns responses by URL suffix."""

    def side_effect(url, api_key, timeout=30):
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        # Default: 404
        return (404, None, {})

    return side_effect


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Test CLI argument parser construction and parsing."""

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://localhost:8000", "--api-key", "test"])
        assert args.base_url == "http://localhost:8000"
        assert args.api_key == "test"

    def test_default_profile(self):
        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://x", "--api-key", "k"])
        assert args.profile == PROFILE_SINGLE_NODE

    def test_profile_choices(self):
        parser = build_parser()
        for profile in ALL_PROFILES:
            args = parser.parse_args(
                ["--base-url", "http://x", "--api-key", "k", "--profile", profile]
            )
            assert args.profile == profile

    def test_invalid_profile_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--base-url", "http://x", "--api-key", "k", "--profile", "invalid"]
            )

    def test_skip_job_submit_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--base-url", "http://x", "--api-key", "k", "--skip-job-submit"]
        )
        assert args.skip_job_submit is True

    def test_default_skip_job_submit(self):
        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://x", "--api-key", "k"])
        assert args.skip_job_submit is False

    def test_timeout_default(self):
        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://x", "--api-key", "k"])
        assert args.timeout == DEFAULT_TIMEOUT_SECONDS

    def test_timeout_custom(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--base-url", "http://x", "--api-key", "k", "--timeout", "120"]
        )
        assert args.timeout == 120

    def test_format_default(self):
        parser = build_parser()
        args = parser.parse_args(["--base-url", "http://x", "--api-key", "k"])
        assert args.output_format == "json"

    def test_format_markdown(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--base-url", "http://x", "--api-key", "k", "--format", "markdown"]
        )
        assert args.output_format == "markdown"

    def test_output_file(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--base-url", "http://x", "--api-key", "k", "--output", "report.json"]
        )
        assert args.output == "report.json"

    def test_missing_required_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Health check logic
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Test health check logic with mock HTTP responses."""

    @patch("scripts.validate_deployment._http_get")
    def test_healthy_response(self, mock_get):
        mock_get.return_value = (200, {"status": "healthy", "version": "1.2.0"}, {})
        result = check_health("http://localhost:8000", "key")
        assert result.passed is True
        assert result.name == "health_check"
        assert "healthy" in result.detail
        assert "1.2.0" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_unhealthy_response(self, mock_get):
        mock_get.return_value = (200, {"status": "unhealthy"}, {})
        result = check_health("http://localhost:8000", "key")
        # 200 with any body is still a "pass" (server responded)
        assert result.passed is True
        assert "unhealthy" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_non_200_response(self, mock_get):
        mock_get.return_value = (503, None, {})
        result = check_health("http://localhost:8000", "key")
        assert result.passed is False
        assert "503" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        result = check_health("http://localhost:8000", "key")
        assert result.passed is False
        assert "refused" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_duration_recorded(self, mock_get):
        mock_get.return_value = (200, {"status": "healthy"}, {})
        result = check_health("http://localhost:8000", "key")
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Detailed health check logic
# ---------------------------------------------------------------------------


class TestDetailedHealthCheck:
    """Test detailed health check logic with mock HTTP responses."""

    @patch("scripts.validate_deployment._http_get")
    def test_healthy_detailed(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "healthy",
                "checks": {
                    "database": {"status": "healthy"},
                    "disk_output": {"status": "healthy"},
                },
            },
            {},
        )
        result = check_detailed_health("http://localhost:8000", "key")
        assert result.passed is True
        assert "overall=healthy" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_degraded_still_passes(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "degraded",
                "checks": {"database": {"status": "degraded"}},
            },
            {},
        )
        result = check_detailed_health("http://localhost:8000", "key")
        assert result.passed is True
        assert "degraded" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_unhealthy_fails(self, mock_get):
        mock_get.return_value = (
            200,
            {
                "status": "unhealthy",
                "checks": {"database": {"status": "unhealthy"}},
            },
            {},
        )
        result = check_detailed_health("http://localhost:8000", "key")
        assert result.passed is False

    @patch("scripts.validate_deployment._http_get")
    def test_non_200(self, mock_get):
        mock_get.return_value = (500, None, {})
        result = check_detailed_health("http://localhost:8000", "key")
        assert result.passed is False
        assert "500" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_empty_body(self, mock_get):
        mock_get.return_value = (200, None, {})
        result = check_detailed_health("http://localhost:8000", "key")
        assert result.passed is False
        assert "Empty" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_multi_gpu_profile(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy", "checks": {}},
            {},
        )
        result = check_detailed_health(
            "http://localhost:8000", "key", profile=PROFILE_MULTI_GPU
        )
        assert result.passed is True

    @patch("scripts.validate_deployment._http_get")
    def test_kubernetes_profile(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy", "checks": {}},
            {},
        )
        result = check_detailed_health(
            "http://localhost:8000", "key", profile=PROFILE_KUBERNETES
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Schema validation logic
# ---------------------------------------------------------------------------


class TestSchemaList:
    """Test schema list check logic."""

    @patch("scripts.validate_deployment._http_get")
    def test_all_schemas_present(self, mock_get):
        schemas = [{"output_type": t, "schema_version": "1.0"} for t in EXPECTED_SCHEMA_TYPES]
        mock_get.return_value = (200, {"schemas": schemas}, {})
        result = check_schema_list("http://localhost:8000", "key")
        assert result.passed is True
        assert "14" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_missing_schemas(self, mock_get):
        # Only return 10 of 14
        schemas = [
            {"output_type": t, "schema_version": "1.0"}
            for t in list(EXPECTED_SCHEMA_TYPES)[:10]
        ]
        mock_get.return_value = (200, {"schemas": schemas}, {})
        result = check_schema_list("http://localhost:8000", "key")
        assert result.passed is False
        assert "Missing" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_non_200(self, mock_get):
        mock_get.return_value = (401, None, {})
        result = check_schema_list("http://localhost:8000", "key")
        assert result.passed is False

    @patch("scripts.validate_deployment._http_get")
    def test_missing_schemas_field(self, mock_get):
        mock_get.return_value = (200, {"something": "else"}, {})
        result = check_schema_list("http://localhost:8000", "key")
        assert result.passed is False
        assert "schemas" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        result = check_schema_list("http://localhost:8000", "key")
        assert result.passed is False
        assert "timeout" in result.detail


class TestSchemaRetrieval:
    """Test individual schema retrieval checks."""

    @patch("scripts.validate_deployment._http_get")
    def test_all_schemas_valid(self, mock_get):
        mock_get.return_value = (
            200,
            {"type": "object", "title": "Test", "properties": {}},
            {},
        )
        results = check_schema_retrieval("http://localhost:8000", "key")
        assert len(results) == len(EXPECTED_SCHEMA_TYPES)
        assert all(r.passed for r in results)

    @patch("scripts.validate_deployment._http_get")
    def test_one_schema_fails(self, mock_get):
        call_count = [0]

        def side_effect(url, api_key, timeout=30):
            call_count[0] += 1
            if call_count[0] == 3:
                return (404, None, {})
            return (200, {"type": "object", "properties": {}}, {})

        mock_get.side_effect = side_effect
        results = check_schema_retrieval("http://localhost:8000", "key")
        assert len(results) == len(EXPECTED_SCHEMA_TYPES)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 1

    @patch("scripts.validate_deployment._http_get")
    def test_invalid_json_schema(self, mock_get):
        # Valid JSON but not a schema (no type/properties/$schema/title)
        mock_get.return_value = (200, {"data": "not a schema"}, {})
        results = check_schema_retrieval("http://localhost:8000", "key")
        assert all(not r.passed for r in results)
        assert "lacks JSON Schema markers" in results[0].detail

    @patch("scripts.validate_deployment._http_get")
    def test_non_dict_response(self, mock_get):
        mock_get.return_value = (200, None, {})
        results = check_schema_retrieval("http://localhost:8000", "key")
        assert all(not r.passed for r in results)

    @patch("scripts.validate_deployment._http_get")
    def test_check_names_include_schema_type(self, mock_get):
        mock_get.return_value = (200, {"type": "object"}, {})
        results = check_schema_retrieval("http://localhost:8000", "key")
        for i, schema_type in enumerate(EXPECTED_SCHEMA_TYPES):
            assert results[i].name == f"schema_get_{schema_type}"


# ---------------------------------------------------------------------------
# Version header check
# ---------------------------------------------------------------------------


class TestVersionHeader:
    """Test version header check logic."""

    @patch("scripts.validate_deployment._http_get")
    def test_version_present(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy"},
            {"x-api-version": "1.2.0"},
        )
        result = check_version_header("http://localhost:8000", "key")
        assert result.passed is True
        assert "1.2.0" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_version_missing(self, mock_get):
        mock_get.return_value = (200, {"status": "healthy"}, {})
        result = check_version_header("http://localhost:8000", "key")
        assert result.passed is False
        assert "not found" in result.detail

    @patch("scripts.validate_deployment._http_get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        result = check_version_header("http://localhost:8000", "key")
        assert result.passed is False
        assert "connection refused" in result.detail


# ---------------------------------------------------------------------------
# Job submission check
# ---------------------------------------------------------------------------


class TestJobSubmission:
    """Test job submission, polling, and manifest verification."""

    @patch("scripts.validate_deployment.POLL_INTERVAL_SECONDS", 0)
    @patch("scripts.validate_deployment._http_get")
    @patch("scripts.validate_deployment._http_post_multipart")
    def test_full_success(self, mock_post, mock_get):
        mock_post.return_value = (201, {"job_id": "job_abc123456789"}, {})
        mock_get.side_effect = [
            # First poll: processing
            (200, {"status": "processing"}, {}),
            # Second poll: completed
            (200, {"status": "completed"}, {}),
            # Manifest retrieval
            (
                200,
                {
                    "job_id": "job_abc123456789",
                    "artifacts": [
                        {"output_type": "searchable_pdf"},
                        {"output_type": "ocr_text"},
                    ],
                },
                {},
            ),
        ]
        results = check_job_submission("http://localhost:8000", "key", timeout=10)
        assert len(results) == 3
        assert results[0].name == "job_submit"
        assert results[0].passed is True
        assert results[1].name == "job_completion"
        assert results[1].passed is True
        assert results[2].name == "job_output_manifest"
        assert results[2].passed is True

    @patch("scripts.validate_deployment._http_post_multipart")
    def test_submit_failure(self, mock_post):
        mock_post.return_value = (500, {"error": "internal"}, {})
        results = check_job_submission("http://localhost:8000", "key")
        assert len(results) == 1
        assert results[0].passed is False
        assert "500" in results[0].detail

    @patch("scripts.validate_deployment._http_post_multipart")
    def test_submit_connection_error(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")
        results = check_job_submission("http://localhost:8000", "key")
        assert len(results) == 1
        assert results[0].passed is False
        assert "refused" in results[0].detail

    @patch("scripts.validate_deployment._http_post_multipart")
    def test_no_job_id_in_response(self, mock_post):
        mock_post.return_value = (201, {"status": "created"}, {})
        results = check_job_submission("http://localhost:8000", "key")
        assert len(results) == 1
        assert results[0].passed is False
        assert "job_id" in results[0].detail.lower()

    @patch("scripts.validate_deployment.POLL_INTERVAL_SECONDS", 0)
    @patch("scripts.validate_deployment._http_get")
    @patch("scripts.validate_deployment._http_post_multipart")
    def test_job_failed(self, mock_post, mock_get):
        mock_post.return_value = (201, {"job_id": "job_abc123456789"}, {})
        mock_get.return_value = (200, {"status": "failed"}, {})
        results = check_job_submission("http://localhost:8000", "key", timeout=5)
        assert len(results) == 2
        assert results[1].name == "job_completion"
        assert results[1].passed is False
        assert "failed" in results[1].detail.lower()

    @patch("scripts.validate_deployment.POLL_INTERVAL_SECONDS", 0)
    @patch("scripts.validate_deployment._http_get")
    @patch("scripts.validate_deployment._http_post_multipart")
    def test_job_timeout(self, mock_post, mock_get):
        mock_post.return_value = (201, {"job_id": "job_abc123456789"}, {})
        mock_get.return_value = (200, {"status": "processing"}, {})
        results = check_job_submission("http://localhost:8000", "key", timeout=0)
        assert len(results) == 2
        assert results[1].name == "job_completion"
        assert results[1].passed is False
        assert "did not complete" in results[1].detail.lower()

    @patch("scripts.validate_deployment.POLL_INTERVAL_SECONDS", 0)
    @patch("scripts.validate_deployment._http_get")
    @patch("scripts.validate_deployment._http_post_multipart")
    def test_manifest_missing_artifacts(self, mock_post, mock_get):
        mock_post.return_value = (201, {"job_id": "job_abc123456789"}, {})
        mock_get.side_effect = [
            (200, {"status": "completed"}, {}),
            (200, {"job_id": "job_abc123456789"}, {}),  # No "artifacts" key
        ]
        results = check_job_submission("http://localhost:8000", "key", timeout=5)
        assert len(results) == 3
        assert results[2].name == "job_output_manifest"
        assert results[2].passed is False
        assert "artifacts" in results[2].detail.lower()


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------


class TestDeploymentReport:
    """Test DeploymentReport data model properties."""

    def test_empty_report(self):
        report = DeploymentReport()
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.all_passed is True  # vacuously true

    def test_all_passed(self):
        report = DeploymentReport(
            checks=[
                CheckResult(name="a", passed=True),
                CheckResult(name="b", passed=True),
            ]
        )
        assert report.total == 2
        assert report.passed == 2
        assert report.failed == 0
        assert report.all_passed is True

    def test_some_failed(self):
        report = DeploymentReport(
            checks=[
                CheckResult(name="a", passed=True),
                CheckResult(name="b", passed=False),
            ]
        )
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1
        assert report.all_passed is False


# ---------------------------------------------------------------------------
# JSON report generation
# ---------------------------------------------------------------------------


class TestJSONReport:
    """Test JSON report formatting."""

    def test_valid_json(self):
        report = DeploymentReport(
            base_url="http://localhost:8000",
            profile=PROFILE_SINGLE_NODE,
            timestamp="2026-03-29T00:00:00+00:00",
            checks=[
                CheckResult(name="health_check", passed=True, detail="OK", duration_ms=5.0),
                CheckResult(name="schema_list", passed=False, detail="Missing types"),
            ],
        )
        output = format_json_report(report)
        data = json.loads(output)
        assert data["base_url"] == "http://localhost:8000"
        assert data["profile"] == "single-node"
        assert data["summary"]["total"] == 2
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["all_passed"] is False
        assert len(data["checks"]) == 2
        assert data["checks"][0]["name"] == "health_check"
        assert data["checks"][0]["passed"] is True
        assert data["checks"][1]["passed"] is False

    def test_empty_report_json(self):
        report = DeploymentReport(
            base_url="http://x",
            profile="single-node",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        output = format_json_report(report)
        data = json.loads(output)
        assert data["summary"]["total"] == 0
        assert data["summary"]["all_passed"] is True

    def test_duration_in_json(self):
        report = DeploymentReport(
            checks=[CheckResult(name="c", passed=True, duration_ms=12.3)]
        )
        data = json.loads(format_json_report(report))
        assert data["checks"][0]["duration_ms"] == 12.3


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    """Test Markdown report formatting."""

    def test_contains_header(self):
        report = DeploymentReport(
            base_url="http://localhost:8000",
            profile=PROFILE_SINGLE_NODE,
            timestamp="2026-03-29T00:00:00+00:00",
        )
        output = format_markdown_report(report)
        assert "# Deployment Validation Report" in output

    def test_contains_metadata(self):
        report = DeploymentReport(
            base_url="http://test:9000",
            profile=PROFILE_MULTI_GPU,
            timestamp="2026-03-29T12:00:00+00:00",
        )
        output = format_markdown_report(report)
        assert "http://test:9000" in output
        assert "multi-gpu" in output
        assert "2026-03-29" in output

    def test_pass_result(self):
        report = DeploymentReport(
            checks=[CheckResult(name="health", passed=True, detail="OK")]
        )
        output = format_markdown_report(report)
        assert "PASS" in output
        assert "1/1 passed" in output

    def test_fail_result(self):
        report = DeploymentReport(
            checks=[CheckResult(name="health", passed=False, detail="Broken")]
        )
        output = format_markdown_report(report)
        assert "FAIL" in output
        assert "0/1 passed" in output

    def test_table_format(self):
        report = DeploymentReport(
            checks=[
                CheckResult(name="check_a", passed=True, detail="good", duration_ms=5.0),
                CheckResult(name="check_b", passed=False, detail="bad", duration_ms=10.0),
            ]
        )
        output = format_markdown_report(report)
        assert "| Check |" in output
        assert "| check_a | PASS |" in output
        assert "| check_b | FAIL |" in output

    def test_pipe_escaped_in_detail(self):
        report = DeploymentReport(
            checks=[CheckResult(name="test", passed=True, detail="a|b")]
        )
        output = format_markdown_report(report)
        assert "a\\|b" in output


# ---------------------------------------------------------------------------
# Topology profile selection
# ---------------------------------------------------------------------------


class TestProfileSelection:
    """Test that profile influences validation behavior."""

    @patch("scripts.validate_deployment._http_get")
    def test_single_node_profile(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy", "checks": {}},
            {},
        )
        result = check_detailed_health(
            "http://localhost:8000", "key", profile=PROFILE_SINGLE_NODE
        )
        assert result.passed is True
        assert result.name == "detailed_health_check"

    @patch("scripts.validate_deployment._http_get")
    def test_multi_gpu_profile_runs(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy", "checks": {"gpu_0": {"status": "healthy"}}},
            {},
        )
        result = check_detailed_health(
            "http://localhost:8000", "key", profile=PROFILE_MULTI_GPU
        )
        assert result.passed is True

    @patch("scripts.validate_deployment._http_get")
    def test_kubernetes_profile_runs(self, mock_get):
        mock_get.return_value = (
            200,
            {"status": "healthy", "checks": {"k8s_pods": {"status": "healthy"}}},
            {},
        )
        result = check_detailed_health(
            "http://localhost:8000", "key", profile=PROFILE_KUBERNETES
        )
        assert result.passed is True

    def test_all_profiles_are_valid(self):
        assert PROFILE_SINGLE_NODE in ALL_PROFILES
        assert PROFILE_MULTI_GPU in ALL_PROFILES
        assert PROFILE_KUBERNETES in ALL_PROFILES
        assert len(ALL_PROFILES) == 3


# ---------------------------------------------------------------------------
# Run validation orchestrator
# ---------------------------------------------------------------------------


class TestRunValidation:
    """Test the run_validation orchestrator."""

    @patch("scripts.validate_deployment._http_get")
    def test_skip_job_submit(self, mock_get):
        # Return healthy for all GET requests
        mock_get.return_value = (
            200,
            {
                "status": "healthy",
                "version": "1.2.0",
                "checks": {},
                "schemas": [
                    {"output_type": t, "schema_version": "1.0"}
                    for t in EXPECTED_SCHEMA_TYPES
                ],
                "type": "object",
                "properties": {},
            },
            {"x-api-version": "1.2.0"},
        )

        report = run_validation(
            base_url="http://localhost:8000",
            api_key="test",
            skip_job_submit=True,
        )

        # Should have: health + detailed_health + schema_list + 14 schema_get + version_header
        expected_count = 1 + 1 + 1 + len(EXPECTED_SCHEMA_TYPES) + 1
        assert report.total == expected_count
        assert report.base_url == "http://localhost:8000"
        assert report.profile == PROFILE_SINGLE_NODE

    @patch("scripts.validate_deployment._http_get")
    def test_includes_timestamp(self, mock_get):
        mock_get.return_value = (200, {"status": "healthy", "type": "object"}, {})
        report = run_validation(
            base_url="http://x", api_key="k", skip_job_submit=True
        )
        assert report.timestamp  # Non-empty
        assert "T" in report.timestamp  # ISO format

    @patch("scripts.validate_deployment._http_get")
    def test_profile_passed_through(self, mock_get):
        mock_get.return_value = (200, {"status": "healthy", "type": "object"}, {})
        report = run_validation(
            base_url="http://x",
            api_key="k",
            profile=PROFILE_KUBERNETES,
            skip_job_submit=True,
        )
        assert report.profile == PROFILE_KUBERNETES


# ---------------------------------------------------------------------------
# CLI main() function
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Test the main() CLI entry point."""

    @patch("scripts.validate_deployment._http_get")
    def test_returns_zero_on_all_pass(self, mock_get, capsys):
        mock_get.return_value = (
            200,
            {
                "status": "healthy",
                "checks": {},
                "schemas": [
                    {"output_type": t, "schema_version": "1.0"}
                    for t in EXPECTED_SCHEMA_TYPES
                ],
                "type": "object",
                "properties": {},
            },
            {"x-api-version": "1.2.0"},
        )
        rc = main([
            "--base-url", "http://localhost:8000",
            "--api-key", "test",
            "--skip-job-submit",
        ])
        assert rc == 0

    @patch("scripts.validate_deployment._http_get")
    def test_returns_one_on_failure(self, mock_get, capsys):
        mock_get.return_value = (500, None, {})
        rc = main([
            "--base-url", "http://localhost:8000",
            "--api-key", "test",
            "--skip-job-submit",
        ])
        assert rc == 1

    @patch("scripts.validate_deployment._http_get")
    def test_json_output_to_stdout(self, mock_get, capsys):
        mock_get.return_value = (
            200,
            {"status": "healthy", "type": "object"},
            {"x-api-version": "1.2.0"},
        )
        main([
            "--base-url", "http://localhost:8000",
            "--api-key", "test",
            "--skip-job-submit",
            "--format", "json",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "summary" in data
        assert "checks" in data

    @patch("scripts.validate_deployment._http_get")
    def test_markdown_output_to_stdout(self, mock_get, capsys):
        mock_get.return_value = (
            200,
            {"status": "healthy", "type": "object"},
            {"x-api-version": "1.2.0"},
        )
        main([
            "--base-url", "http://localhost:8000",
            "--api-key", "test",
            "--skip-job-submit",
            "--format", "markdown",
        ])
        captured = capsys.readouterr()
        assert "# Deployment Validation Report" in captured.out

    @patch("scripts.validate_deployment._http_get")
    def test_output_to_file(self, mock_get, tmp_path):
        mock_get.return_value = (
            200,
            {"status": "healthy", "type": "object"},
            {"x-api-version": "1.2.0"},
        )
        output_file = str(tmp_path / "report.json")
        main([
            "--base-url", "http://localhost:8000",
            "--api-key", "test",
            "--skip-job-submit",
            "--output", output_file,
        ])
        assert os.path.isfile(output_file)
        with open(output_file) as f:
            data = json.loads(f.read())
        assert "summary" in data

    @patch("scripts.validate_deployment._http_get")
    def test_trailing_slash_stripped(self, mock_get, capsys):
        mock_get.return_value = (
            200,
            {"status": "healthy", "type": "object"},
            {"x-api-version": "1.2.0"},
        )
        main([
            "--base-url", "http://localhost:8000/",
            "--api-key", "test",
            "--skip-job-submit",
        ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["base_url"] == "http://localhost:8000"


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constant values are consistent."""

    def test_expected_schema_types_count(self):
        assert len(EXPECTED_SCHEMA_TYPES) == 14

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT_SECONDS == 60

    def test_profiles_list(self):
        assert len(ALL_PROFILES) == 3

    def test_schema_types_match_schemas_package(self):
        """Verify our constants match the schemas/ package output types."""
        try:
            from schemas import OUTPUT_TYPES
            assert set(EXPECTED_SCHEMA_TYPES) == set(OUTPUT_TYPES)
        except ImportError:
            pytest.skip("schemas package not importable")
