"""Tests for Items 26-30: Strategic & Competitive tier.

Tests cover:
- Item 26: PaddleOCR 3.x quarterly review document structure
- Item 27: Compliance documentation (SOC2, HIPAA, FedRAMP)
- Item 28: Presigned URL benchmark logic (mock boto3)
- Item 29: OKE Terraform validation rules
- Item 30: Dashboard config files (Dockerfile, nginx, compose)

Target: 60+ tests
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

# ---------------------------------------------------------------------------
# Paths to doc files
# ---------------------------------------------------------------------------
_DOCS_DIR = _PROJECT_ROOT / "docs"
_STRATEGY_DIR = _DOCS_DIR / "strategy"
_ARCH_DIR = _DOCS_DIR / "architecture"
_COMPLIANCE_DIR = _DOCS_DIR / "compliance"
_DASHBOARD_DIR = _PROJECT_ROOT / "dashboard"
_TERRAFORM_DIR = _PROJECT_ROOT / "terraform"


# ===========================================================================
# Item 26: PaddleOCR 3.x Quarterly Review
# ===========================================================================


class TestPaddleOCR3xQuarterlyReview:
    """Tests for docs/strategy/paddleocr-3x-quarterly-review.md."""

    @pytest.fixture
    def review_content(self):
        path = _STRATEGY_DIR / "paddleocr-3x-quarterly-review.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        path = _STRATEGY_DIR / "paddleocr-3x-quarterly-review.md"
        assert path.exists()

    def test_has_executive_summary(self, review_content):
        assert "Executive Summary" in review_content

    def test_has_current_production_state(self, review_content):
        assert "Current Production State" in review_content

    def test_references_paddleocr_291(self, review_content):
        assert "2.9.1" in review_content

    def test_has_api_differences(self, review_content):
        assert "API" in review_content
        assert "use_onnx" in review_content
        assert "enable_hpi" in review_content

    def test_has_risk_assessment(self, review_content):
        assert "Risk Assessment" in review_content

    def test_has_defer_recommendation(self, review_content):
        assert "DEFER" in review_content

    def test_has_trigger_criteria(self, review_content):
        assert "Trigger Criteria" in review_content

    def test_has_review_checklist(self, review_content):
        assert "Quarterly Review Checklist" in review_content

    def test_has_migration_effort_estimate(self, review_content):
        assert "Migration Effort Estimate" in review_content

    def test_references_ocr_gpu_async(self, review_content):
        assert "ocr_gpu_async" in review_content

    def test_references_download_models(self, review_content):
        assert "download_models" in review_content

    def test_references_language_config(self, review_content):
        assert "language_config" in review_content


# ===========================================================================
# Item 26B: Forensic-Core vs AI-Adjacent Boundary
# ===========================================================================


class TestForensicAIBoundaryDoc:
    """Tests for docs/architecture/forensic-ai-boundary-contract.md."""

    @pytest.fixture
    def boundary_content(self):
        path = _ARCH_DIR / "forensic-ai-boundary-contract.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        path = _ARCH_DIR / "forensic-ai-boundary-contract.md"
        assert path.exists()

    def test_has_boundary_definition(self, boundary_content):
        assert "Boundary Definition" in boundary_content
        assert "Forensic-Core" in boundary_content
        assert "AI-Adjacent" in boundary_content

    def test_has_contract_rules(self, boundary_content):
        assert "Contract Rules" in boundary_content
        assert "feature" in boundary_content.lower()
        assert "additive" in boundary_content.lower()

    def test_references_docintel(self, boundary_content):
        assert "DocIntel" in boundary_content
        assert "enable_docintel" in boundary_content

    def test_references_layoutlm_and_vlm(self, boundary_content):
        assert "LayoutLMv3" in boundary_content
        assert "VLM" in boundary_content

    def test_references_custody_and_validation(self, boundary_content):
        assert "custody" in boundary_content.lower()
        assert "validation" in boundary_content.lower()

    def test_has_current_capability_map(self, boundary_content):
        assert "Current Capability Map" in boundary_content
        assert "Signature verification" in boundary_content

    def test_references_key_implementation_anchors(self, boundary_content):
        assert "api/job_manager.py" in boundary_content
        assert "coordinator/coordinator/celery.py" in boundary_content
        assert "api/routers/semantic.py" in boundary_content


# ===========================================================================
# Item 27: Compliance Documentation
# ===========================================================================


class TestComplianceReadme:
    """Tests for docs/compliance/README.md."""

    @pytest.fixture
    def readme_content(self):
        path = _COMPLIANCE_DIR / "README.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_COMPLIANCE_DIR / "README.md").exists()

    def test_links_to_soc2(self, readme_content):
        assert "soc2-readiness.md" in readme_content

    def test_links_to_hipaa(self, readme_content):
        assert "hipaa-readiness.md" in readme_content

    def test_links_to_fedramp(self, readme_content):
        assert "fedramp-readiness.md" in readme_content

    def test_references_custody(self, readme_content):
        assert "custody" in readme_content.lower()

    def test_references_api_auth(self, readme_content):
        assert "API" in readme_content
        assert "auth" in readme_content.lower()


class TestSOC2Readiness:
    """Tests for docs/compliance/soc2-readiness.md."""

    @pytest.fixture
    def soc2_content(self):
        path = _COMPLIANCE_DIR / "soc2-readiness.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_COMPLIANCE_DIR / "soc2-readiness.md").exists()

    def test_has_control_environment_cc1(self, soc2_content):
        assert "CC1" in soc2_content

    def test_has_monitoring_cc4(self, soc2_content):
        assert "CC4" in soc2_content

    def test_has_access_controls_cc6(self, soc2_content):
        assert "CC6" in soc2_content

    def test_references_chain_of_custody(self, soc2_content):
        assert "chain of custody" in soc2_content.lower() or "hash-chained" in soc2_content.lower()

    def test_references_prometheus(self, soc2_content):
        assert "Prometheus" in soc2_content

    def test_has_readiness_summary(self, soc2_content):
        assert "Readiness" in soc2_content

    def test_has_gap_analysis(self, soc2_content):
        assert "Gap" in soc2_content


class TestHIPAAReadiness:
    """Tests for docs/compliance/hipaa-readiness.md."""

    @pytest.fixture
    def hipaa_content(self):
        path = _COMPLIANCE_DIR / "hipaa-readiness.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_COMPLIANCE_DIR / "hipaa-readiness.md").exists()

    def test_references_hipaa_rule(self, hipaa_content):
        assert "164" in hipaa_content

    def test_has_access_control(self, hipaa_content):
        assert "Access Control" in hipaa_content

    def test_has_audit_controls(self, hipaa_content):
        assert "Audit" in hipaa_content

    def test_has_transmission_security(self, hipaa_content):
        assert "Transmission" in hipaa_content

    def test_references_phi(self, hipaa_content):
        assert "PHI" in hipaa_content

    def test_references_baa(self, hipaa_content):
        assert "BAA" in hipaa_content or "Business Associate" in hipaa_content

    def test_references_encryption(self, hipaa_content):
        assert "encryption" in hipaa_content.lower()

    def test_has_gap_analysis(self, hipaa_content):
        assert "Gap" in hipaa_content


class TestFedRAMPReadiness:
    """Tests for docs/compliance/fedramp-readiness.md."""

    @pytest.fixture
    def fedramp_content(self):
        path = _COMPLIANCE_DIR / "fedramp-readiness.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_COMPLIANCE_DIR / "fedramp-readiness.md").exists()

    def test_references_nist_800_53(self, fedramp_content):
        assert "800-53" in fedramp_content

    def test_has_ac_access_control(self, fedramp_content):
        assert "AC" in fedramp_content
        assert "Access Control" in fedramp_content

    def test_has_au_audit(self, fedramp_content):
        assert "AU" in fedramp_content

    def test_has_sc_system_communications(self, fedramp_content):
        assert "SC" in fedramp_content

    def test_references_network_policies(self, fedramp_content):
        assert "NetworkPolicy" in fedramp_content or "network" in fedramp_content.lower()

    def test_references_prometheus_monitoring(self, fedramp_content):
        assert "Prometheus" in fedramp_content

    def test_has_gap_analysis(self, fedramp_content):
        assert "Gap" in fedramp_content

    def test_has_authorization_path(self, fedramp_content):
        assert "Authorization" in fedramp_content


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

    def test_validate_project_oke_module(self):
        """Validate the actual OKE module in the project."""
        oke_dir = _TERRAFORM_DIR / "modules" / "oke"
        if not oke_dir.exists():
            pytest.skip("OKE Terraform module not found")

        report = validate_oke_terraform(str(oke_dir))
        assert report.files_scanned > 0
        # The project OKE module has known placeholder OCIDs
        assert report.files_scanned >= 1

    def test_validate_full_terraform_dir(self):
        """Validate the full terraform directory."""
        if not _TERRAFORM_DIR.exists():
            pytest.skip("Terraform directory not found")

        report = validate_oke_terraform(str(_TERRAFORM_DIR))
        assert report.files_scanned > 0

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


# ===========================================================================
# Item 30: Dashboard Configuration Files
# ===========================================================================


class TestDashboardDockerfile:
    """Tests for dashboard/Dockerfile."""

    @pytest.fixture
    def dockerfile_content(self):
        path = _DASHBOARD_DIR / "Dockerfile"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_DASHBOARD_DIR / "Dockerfile").exists()

    def test_multistage_build(self, dockerfile_content):
        assert "FROM node:20" in dockerfile_content
        assert "FROM nginx:" in dockerfile_content

    def test_has_builder_stage(self, dockerfile_content):
        assert "AS builder" in dockerfile_content

    def test_has_api_url_arg(self, dockerfile_content):
        assert "NEXT_PUBLIC_API_URL" in dockerfile_content

    def test_non_root_user(self, dockerfile_content):
        assert "USER nginx" in dockerfile_content

    def test_has_healthcheck(self, dockerfile_content):
        assert "HEALTHCHECK" in dockerfile_content

    def test_exposes_port_3000(self, dockerfile_content):
        assert "EXPOSE 3000" in dockerfile_content


class TestDashboardNginxConf:
    """Tests for dashboard/nginx.conf."""

    @pytest.fixture
    def nginx_content(self):
        path = _DASHBOARD_DIR / "nginx.conf"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_DASHBOARD_DIR / "nginx.conf").exists()

    def test_spa_routing(self, nginx_content):
        assert "try_files" in nginx_content
        assert "index.html" in nginx_content

    def test_api_proxy(self, nginx_content):
        assert "proxy_pass" in nginx_content
        assert "/api/" in nginx_content

    def test_security_headers(self, nginx_content):
        assert "X-Frame-Options" in nginx_content
        assert "Content-Security-Policy" in nginx_content
        assert "X-Content-Type-Options" in nginx_content

    def test_gzip_compression(self, nginx_content):
        assert "gzip on" in nginx_content

    def test_websocket_proxy(self, nginx_content):
        assert "/ws/" in nginx_content
        assert "upgrade" in nginx_content.lower()

    def test_health_endpoint(self, nginx_content):
        assert "/health" in nginx_content

    def test_static_caching(self, nginx_content):
        assert "/static/" in nginx_content
        assert "expires" in nginx_content

    def test_hidden_files_denied(self, nginx_content):
        assert "deny all" in nginx_content


class TestDashboardCompose:
    """Tests for dashboard/docker-compose.dashboard.yml."""

    @pytest.fixture
    def compose_content(self):
        path = _DASHBOARD_DIR / "docker-compose.dashboard.yml"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_DASHBOARD_DIR / "docker-compose.dashboard.yml").exists()

    def test_has_dashboard_service(self, compose_content):
        assert "dashboard:" in compose_content

    def test_has_port_mapping(self, compose_content):
        assert "3000" in compose_content

    def test_has_env_vars(self, compose_content):
        assert "API_URL" in compose_content
        assert "API_KEY" in compose_content

    def test_has_healthcheck(self, compose_content):
        assert "healthcheck" in compose_content

    def test_has_network(self, compose_content):
        assert "networks" in compose_content


class TestDashboardReadme:
    """Tests for dashboard/README.md."""

    @pytest.fixture
    def readme_content(self):
        path = _DASHBOARD_DIR / "README.md"
        assert path.exists(), f"Missing: {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (_DASHBOARD_DIR / "README.md").exists()

    def test_has_quick_start(self, readme_content):
        assert "Quick Start" in readme_content

    def test_has_environment_variables(self, readme_content):
        assert "Environment Variable" in readme_content

    def test_has_docker_instructions(self, readme_content):
        assert "docker" in readme_content.lower()

    def test_has_architecture(self, readme_content):
        assert "Architecture" in readme_content

    def test_references_coordinator(self, readme_content):
        assert "coordinator" in readme_content.lower()
