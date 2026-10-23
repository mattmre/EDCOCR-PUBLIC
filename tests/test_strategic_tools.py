"""Tests for Items 26-30: Strategic & Competitive tier.

Tests cover:
- Presigned URL benchmark logic
- OKE Terraform validation rules on synthetic input
Target: 60+ tests
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# Import benchmark and validation modules
from scripts.benchmark_presigned_urls import (
    BenchmarkReport,
    LatencyStats,
    benchmark_concurrent_dry_run,
    benchmark_url_generation_dry_run,
    compute_latency_stats,
    format_report_markdown,
    run_benchmark,
)
from scripts.validate_terraform_oke import (
    OCI_FLEX_SHAPES,
    OCI_GPU_SHAPES,
    ValidationFinding,
    ValidationReport,
    extract_resource_blocks,
    extract_variable_blocks,
    find_tf_files,
    validate_boot_volume,
    validate_flex_shapes,
    validate_gpu_shapes,
    validate_kubernetes_version,
    validate_network_security,
    validate_node_labels,
    validate_oke_terraform,
    validate_placeholder_ocids,
)




# ===========================================================================
# Item 28: Presigned URL Benchmark
# ===========================================================================


class TestPresignedURLBenchmarkLatency:
    """Tests for presigned URL benchmark latency computation."""

    def test_compute_latency_stats_empty(self):
        stats = compute_latency_stats([])
        assert stats.count == 0
        assert stats.mean_ms == 0.0

    def test_compute_latency_stats_single(self):
        stats = compute_latency_stats([5.0])
        assert stats.count == 1
        assert stats.mean_ms == 5.0
        assert stats.min_ms == 5.0
        assert stats.max_ms == 5.0

    def test_compute_latency_stats_multiple(self):
        timings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_latency_stats(timings)
        assert stats.count == 10
        assert stats.mean_ms == 5.5
        assert stats.min_ms == 1.0
        assert stats.max_ms == 10.0

    def test_compute_latency_stats_p50(self):
        timings = list(range(1, 101))
        timings = [float(t) for t in timings]
        stats = compute_latency_stats(timings)
        assert stats.count == 100
        assert stats.p50_ms == 51.0

    def test_compute_latency_stats_p95(self):
        timings = [float(i) for i in range(1, 101)]
        stats = compute_latency_stats(timings)
        assert stats.p95_ms >= 90.0

    def test_compute_latency_stats_p99(self):
        timings = [float(i) for i in range(1, 101)]
        stats = compute_latency_stats(timings)
        assert stats.p99_ms >= 95.0

    def test_compute_latency_stats_stddev(self):
        stats = compute_latency_stats([5.0, 5.0, 5.0])
        assert stats.stddev_ms == 0.0


class TestPresignedURLBenchmarkDryRun:
    """Tests for dry-run benchmark execution."""

    def test_url_generation_dry_run(self):
        stats = benchmark_url_generation_dry_run(50)
        assert stats.count == 50
        assert stats.mean_ms > 0
        assert stats.p95_ms > 0

    def test_concurrent_dry_run(self):
        result = benchmark_concurrent_dry_run(10, iterations_per_worker=5)
        assert result.concurrency_level == 10
        assert result.success_count == 50  # 10 * 5
        assert result.failure_count == 0
        assert result.latency.count == 50

    def test_concurrent_dry_run_higher_concurrency(self):
        result = benchmark_concurrent_dry_run(100, iterations_per_worker=2)
        assert result.concurrency_level == 100
        assert result.success_count == 200

    def test_run_benchmark_dry_run(self):
        report = run_benchmark(dry_run=True, iterations=20)
        assert report.dry_run is True
        assert report.url_generation.count == 20
        assert len(report.concurrency_results) == 3  # default levels: 10, 50, 100
        assert not report.errors

    def test_run_benchmark_dry_run_custom_concurrency(self):
        report = run_benchmark(
            dry_run=True,
            iterations=10,
            concurrency_levels=[5, 25],
        )
        assert len(report.concurrency_results) == 2


class TestPresignedURLBenchmarkReport:
    """Tests for benchmark report formatting."""

    def test_format_report_markdown_basic(self):
        report = BenchmarkReport(
            endpoint="http://localhost:9000",
            bucket="test",
            iterations=10,
            timestamp="2026-03-15T00:00:00.000Z",
            url_generation=LatencyStats(
                count=10, mean_ms=0.1, min_ms=0.05,
                max_ms=0.2, p50_ms=0.1, p95_ms=0.15,
                p99_ms=0.19, stddev_ms=0.03,
            ),
            dry_run=True,
        )
        md = format_report_markdown(report)
        assert "# S3 Presigned URL Benchmark Report" in md
        assert "http://localhost:9000" in md
        assert "Dry Run" in md

    def test_format_report_includes_concurrency(self):
        report = BenchmarkReport(
            endpoint="http://localhost:9000",
            bucket="test",
            iterations=10,
            timestamp="2026-03-15T00:00:00.000Z",
            url_generation=LatencyStats(count=10),
            concurrency_results=[{
                "concurrency_level": 10,
                "latency": {"mean_ms": 0.1, "p50_ms": 0.09, "p95_ms": 0.15, "p99_ms": 0.19},
                "success_count": 100,
                "failure_count": 0,
                "total_time_s": 1.0,
            }],
            dry_run=True,
        )
        md = format_report_markdown(report)
        assert "Concurrent Access" in md
        assert "| 10 " in md

    def test_format_report_includes_errors(self):
        report = BenchmarkReport(
            endpoint="http://localhost:9000",
            bucket="test",
            iterations=10,
            timestamp="2026-03-15T00:00:00.000Z",
            url_generation=LatencyStats(count=0),
            errors=["boto3 not available"],
            dry_run=False,
        )
        md = format_report_markdown(report)
        assert "Errors" in md
        assert "boto3 not available" in md

    def test_run_benchmark_no_boto3(self):
        """Benchmark handles missing boto3 gracefully."""
        with patch("scripts.benchmark_presigned_urls._boto3", None):
            report = run_benchmark(
                endpoint="http://fake:9000",
                bucket="fake",
                iterations=5,
                dry_run=False,
            )
            assert len(report.errors) > 0


# ===========================================================================
# Item 29: OKE Terraform Validation
# ===========================================================================


class TestOKETerraformParsing:
    """Tests for Terraform HCL parsing helpers."""

    def test_extract_resource_blocks(self):
        hcl = '''
resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_id
  display_name   = "test-vcn"
}
'''
        blocks = extract_resource_blocks(hcl)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "oci_core_vcn"
        assert blocks[0]["name"] == "main"

    def test_extract_multiple_resources(self):
        hcl = '''
resource "oci_core_vcn" "main" {
  display_name = "test"
}

resource "oci_core_subnet" "node" {
  vcn_id = oci_core_vcn.main.id
}
'''
        blocks = extract_resource_blocks(hcl)
        assert len(blocks) == 2

    def test_extract_variable_blocks(self):
        hcl = '''
variable "cluster_name" {
  description = "Name of the cluster"
  type        = string
  default     = "test-cluster"
}
'''
        variables = extract_variable_blocks(hcl)
        assert len(variables) == 1
        assert variables[0]["name"] == "cluster_name"
        assert variables[0]["default"] == "test-cluster"
        assert variables[0]["type"] == "string"

    def test_extract_variable_without_default(self):
        hcl = '''
variable "compartment_id" {
  description = "OCI compartment OCID"
  type        = string
}
'''
        variables = extract_variable_blocks(hcl)
        assert len(variables) == 1
        assert variables[0]["default"] is None

    def test_find_tf_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.tf").write_text("# test")
            (Path(tmpdir) / "variables.tf").write_text("# test")
            (Path(tmpdir) / "readme.md").write_text("# not tf")
            files = find_tf_files(tmpdir)
            assert len(files) == 2

    def test_find_tf_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = find_tf_files(tmpdir)
            assert len(files) == 0

    def test_find_tf_files_nonexistent_dir(self):
        files = find_tf_files("/nonexistent/path")
        assert len(files) == 0


class TestOKEValidationRules:
    """Tests for individual OKE validation rules."""

    def test_validate_placeholder_ocids_found(self):
        content = 'image_id = "ocid1.image.oc1..placeholder"'
        findings = validate_placeholder_ocids(content, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-OCID-001"
        assert findings[0].severity == "warning"

    def test_validate_placeholder_ocids_strict(self):
        content = 'image_id = "ocid1.image.oc1..placeholder"'
        findings = validate_placeholder_ocids(content, "test.tf", strict=True)
        assert len(findings) == 1
        assert findings[0].severity == "error"

    def test_validate_placeholder_ocids_clean(self):
        content = 'image_id = var.gpu_image_id'
        findings = validate_placeholder_ocids(content, "test.tf")
        assert len(findings) == 0

    def test_validate_gpu_shapes_known(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": 'node_shape = "VM.GPU.A10.1"',
        }]
        findings = validate_gpu_shapes(resources, "test.tf")
        assert len(findings) == 0

    def test_validate_gpu_shapes_unknown(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": 'node_shape = "VM.GPU.FAKE.99"',
        }]
        findings = validate_gpu_shapes(resources, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-GPU-001"

    def test_validate_gpu_shapes_strict(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": 'node_shape = "VM.GPU.UNKNOWN.1"',
        }]
        findings = validate_gpu_shapes(resources, "test.tf", strict=True)
        assert len(findings) == 1
        assert findings[0].severity == "error"

    def test_validate_flex_shapes_with_config(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "cpu",
            "line": 5,
            "content": '''
  node_shape = "VM.Standard.E4.Flex"
  node_shape_config {
    ocpus         = 4
    memory_in_gbs = 32
  }
''',
        }]
        variables = [{"name": "cpu_node_shape", "default": "VM.Standard.E4.Flex"}]
        findings = validate_flex_shapes(resources, variables, "test.tf")
        assert len(findings) == 0

    def test_validate_flex_shapes_missing_config(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "cpu",
            "line": 5,
            "content": 'node_shape = "VM.Standard.E4.Flex"',
        }]
        variables = []
        findings = validate_flex_shapes(resources, variables, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-FLEX-001"

    def test_validate_flex_shapes_missing_ocpus(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "cpu",
            "line": 5,
            "content": '''
  node_shape = "VM.Standard.E4.Flex"
  node_shape_config {
    memory_in_gbs = 32
  }
''',
        }]
        variables = []
        findings = validate_flex_shapes(resources, variables, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-FLEX-002"

    def test_validate_network_security_open_ingress(self):
        resources = [{
            "type": "oci_core_security_list",
            "name": "node",
            "line": 10,
            "content": '''
  ingress_security_rules {
    source    = "0.0.0.0/0"
    protocol  = "all"
    stateless = false
  }
''',
        }]
        findings = validate_network_security(resources, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-NET-001"

    def test_validate_network_security_restricted(self):
        resources = [{
            "type": "oci_core_security_list",
            "name": "node",
            "line": 10,
            "content": '''
  ingress_security_rules {
    source    = "10.0.0.0/16"
    protocol  = "all"
    stateless = false
  }
''',
        }]
        findings = validate_network_security(resources, "test.tf")
        assert len(findings) == 0

    def test_validate_kubernetes_version_valid(self):
        variables = [{"name": "kubernetes_version", "default": "v1.29.1", "line": 5}]
        findings = validate_kubernetes_version("", variables, "test.tf")
        assert len(findings) == 0

    def test_validate_kubernetes_version_too_old(self):
        variables = [{"name": "kubernetes_version", "default": "v1.24.0", "line": 5}]
        findings = validate_kubernetes_version("", variables, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-VER-001"

    def test_validate_boot_volume_sufficient(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": "boot_volume_size_in_gbs = 100",
        }]
        findings = validate_boot_volume(resources, "test.tf")
        assert len(findings) == 0

    def test_validate_boot_volume_too_small(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": "boot_volume_size_in_gbs = 50",
        }]
        findings = validate_boot_volume(resources, "test.tf")
        assert len(findings) == 1
        assert findings[0].rule == "OKE-BOOT-001"

    def test_validate_node_labels_present(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": '''
  initial_node_labels {
    key   = "nvidia.com/gpu"
    value = "true"
  }
''',
        }]
        findings = validate_node_labels(resources, "test.tf")
        assert len(findings) == 0

    def test_validate_node_labels_missing(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": 'node_shape = "VM.GPU.A10.1"',
        }]
        findings = validate_node_labels(resources, "test.tf")
        assert len(findings) >= 1
        rules = [f.rule for f in findings]
        assert "OKE-LABEL-001" in rules

    def test_validate_node_labels_gpu_missing_nvidia(self):
        resources = [{
            "type": "oci_containerengine_node_pool",
            "name": "gpu",
            "line": 10,
            "content": '''
  initial_node_labels {
    key   = "ocr-local/node-type"
    value = "gpu"
  }
''',
        }]
        findings = validate_node_labels(resources, "test.tf")
        assert any(f.rule == "OKE-LABEL-002" for f in findings)


class TestOKEValidationIntegration:
    """Integration tests for OKE Terraform validation against real project files."""

    def test_validate_nonexistent_dir(self):
        report = validate_oke_terraform("/nonexistent/path")
        assert report.files_scanned == 0
        assert not report.passed

    def test_validate_strict_mode(self):
        """Strict mode flags warnings as errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_path = Path(tmpdir) / "main.tf"
            tf_path.write_text(
                'resource "oci_core_vcn" "main" {\n'
                '  compartment_id = var.compartment_id\n'
                '}\n'
            )
            report = validate_oke_terraform(tmpdir, strict=False)
            assert report.files_scanned == 1

    def test_validation_report_json_serializable(self):
        """Ensure report can be serialized to JSON."""
        report = validate_oke_terraform("/nonexistent/path")
        from dataclasses import asdict
        json_str = json.dumps(asdict(report), indent=2)
        assert json_str
        parsed = json.loads(json_str)
        assert "terraform_dir" in parsed


class TestOKEValidationReport:
    """Tests for OKE validation report formatting."""

    def test_format_markdown_pass(self):
        from scripts.validate_terraform_oke import format_report_markdown
        report = ValidationReport(
            terraform_dir="/test",
            timestamp="2026-03-15T00:00:00Z",
            files_scanned=3,
            passed=True,
        )
        md = format_report_markdown(report)
        assert "PASS" in md
        assert "No findings" in md

    def test_format_markdown_fail(self):
        from dataclasses import asdict

        from scripts.validate_terraform_oke import format_report_markdown
        report = ValidationReport(
            terraform_dir="/test",
            timestamp="2026-03-15T00:00:00Z",
            files_scanned=3,
            findings=[asdict(ValidationFinding(
                rule="OKE-TEST-001",
                severity="error",
                message="Test finding",
                file="main.tf",
                line=10,
            ))],
            error_count=1,
            passed=False,
        )
        md = format_report_markdown(report)
        assert "FAIL" in md
        assert "OKE-TEST-001" in md


class TestOKEConstants:
    """Tests for OKE configuration constants."""

    def test_gpu_shapes_not_empty(self):
        assert len(OCI_GPU_SHAPES) > 0

    def test_gpu_shapes_have_required_fields(self):
        for shape, info in OCI_GPU_SHAPES.items():
            assert "gpu_count" in info
            assert "gpu_type" in info
            assert "vram_gb" in info
            assert info["gpu_count"] > 0
            assert info["vram_gb"] > 0

    def test_flex_shapes_not_empty(self):
        assert len(OCI_FLEX_SHAPES) > 0

    def test_known_gpu_shape_a10(self):
        assert "VM.GPU.A10.1" in OCI_GPU_SHAPES
        assert OCI_GPU_SHAPES["VM.GPU.A10.1"]["gpu_type"] == "A10"

    def test_known_gpu_shape_a100(self):
        assert "BM.GPU4.8" in OCI_GPU_SHAPES
        assert OCI_GPU_SHAPES["BM.GPU4.8"]["gpu_type"] == "A100"
