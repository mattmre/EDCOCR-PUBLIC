"""Tests for GKE-specific Terraform validation (scripts/validate_terraform_gke.py).

Covers:
- HCL parsing helpers (read_tf_files, find_resources, extract_values)
- Block extraction and nested block detection
- Cluster validation rules (workload identity, VPC-native, private cluster)
- Node pool validation (GPU types, machine types, autoscaling, taints)
- IAM binding validation (least-privilege, overly-permissive roles)
- Networking validation (VPC, subnet, NAT, firewall)
- Label validation (required and recommended labels)
- Artifact Registry validation
- Autopilot vs Standard mode detection
- Service account validation
- Report generation (JSON and Markdown)
- CLI argument handling
- Strict mode escalation
- Edge cases (empty files, missing blocks)

Run with: python -m pytest tests/test_terraform_gke_validation.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.validate_terraform_gke import (
    Finding,
    Severity,
    ValidationReport,
    _extract_block,
    extract_bool_values,
    extract_number_values,
    extract_string_values,
    find_blocks,
    find_resources,
    has_block,
    main,
    read_tf_files,
    validate_artifact_registry,
    validate_autopilot_detection,
    validate_gke_cluster,
    validate_gke_terraform,
    validate_iam_bindings,
    validate_labels,
    validate_networking,
    validate_node_pools,
    validate_service_account,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_GKE_CLUSTER = """
resource "google_container_cluster" "main" {
  name     = "ocr-local"
  project  = "my-project"
  location = "us-central1"

  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  workload_identity_config {
    workload_pool = "my-project.svc.id.goog"
  }

  resource_labels = var.labels
}
"""

GOOD_GPU_NODE_POOL = """
resource "google_container_node_pool" "gpu" {
  name     = "ocr-gpu"
  project  = "my-project"
  location = "us-central1"
  cluster  = "main"

  autoscaling {
    min_node_count = 0
    max_node_count = 8
  }

  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 200

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "ocr-local/node-type" = "gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
"""

GOOD_CPU_NODE_POOL = """
resource "google_container_node_pool" "cpu" {
  name     = "ocr-cpu"
  project  = "my-project"
  location = "us-central1"
  cluster  = "main"

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  node_config {
    machine_type = "e2-standard-8"

    labels = {
      "ocr-local/node-type" = "cpu"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
"""

GOOD_NETWORKING = """
resource "google_compute_network" "main" {
  name                    = "ocr-vpc"
  project                 = "my-project"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "ocr-subnet"
  project       = "my-project"
  region        = "us-central1"
  network       = google_compute_network.main.id
  ip_cidr_range = "10.0.0.0/20"

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.4.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.8.0.0/20"
  }

  private_ip_google_access = true
}

resource "google_compute_router" "main" {
  name    = "ocr-router"
  project = "my-project"
  region  = "us-central1"
  network = google_compute_network.main.id
}

resource "google_compute_router_nat" "main" {
  name                               = "ocr-nat"
  project                            = "my-project"
  region                             = "us-central1"
  router                             = google_compute_router.main.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
"""

GOOD_IAM = """
resource "google_service_account" "gke_nodes" {
  account_id   = "ocr-nodes"
  project      = "my-project"
  display_name = "GKE node SA"
}

resource "google_project_iam_member" "log_writer" {
  project = "my-project"
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:ocr-nodes@my-project.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "metric_writer" {
  project = "my-project"
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:ocr-nodes@my-project.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "ar_reader" {
  project = "my-project"
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:ocr-nodes@my-project.iam.gserviceaccount.com"
}
"""

GOOD_ARTIFACT_REGISTRY = """
resource "google_artifact_registry_repository" "main" {
  repository_id = "ocr-local"
  project       = "my-project"
  location      = "us-central1"
  format        = "DOCKER"
  description   = "Container images for OCR-Local pipeline"
}
"""

LABELS_VARIABLE = """
variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project    = "ocr-local"
    managed-by = "terraform"
  }
}
"""


@pytest.fixture
def good_tf_dir(tmp_path):
    """Create a temp directory with a well-configured GKE terraform setup."""
    gke_dir = tmp_path / "modules" / "gke"
    gke_dir.mkdir(parents=True)
    (gke_dir / "main.tf").write_text(
        GOOD_GKE_CLUSTER
        + GOOD_GPU_NODE_POOL
        + GOOD_CPU_NODE_POOL
        + GOOD_NETWORKING
        + GOOD_IAM
        + GOOD_ARTIFACT_REGISTRY,
        encoding="utf-8",
    )
    (gke_dir / "variables.tf").write_text(LABELS_VARIABLE, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def empty_tf_dir(tmp_path):
    """Create an empty temp directory."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests: HCL Parsing Helpers
# ---------------------------------------------------------------------------


class TestExtractBlock:
    def test_simple_block(self):
        content = '{ key = "value" }'
        result = _extract_block(content, 1)
        assert 'key = "value"' in result

    def test_nested_blocks(self):
        content = '{ outer { inner = true } }'
        result = _extract_block(content, 1)
        assert "outer" in result
        assert "inner = true" in result

    def test_deeply_nested(self):
        content = '{ a { b { c = 1 } } }'
        result = _extract_block(content, 1)
        assert "c = 1" in result


class TestFindResources:
    def test_finds_cluster(self):
        results = find_resources(GOOD_GKE_CLUSTER, "google_container_cluster")
        assert len(results) == 1
        assert results[0][0] == "main"

    def test_finds_node_pools(self):
        content = GOOD_GPU_NODE_POOL + GOOD_CPU_NODE_POOL
        results = find_resources(content, "google_container_node_pool")
        assert len(results) == 2
        names = {r[0] for r in results}
        assert "gpu" in names
        assert "cpu" in names

    def test_no_match(self):
        results = find_resources(GOOD_GKE_CLUSTER, "google_container_node_pool")
        assert len(results) == 0


class TestFindBlocks:
    def test_finds_workload_identity(self):
        blocks = find_blocks(GOOD_GKE_CLUSTER, "workload_identity_config")
        assert len(blocks) == 1
        assert "workload_pool" in blocks[0]

    def test_finds_autoscaling(self):
        blocks = find_blocks(GOOD_GPU_NODE_POOL, "autoscaling")
        assert len(blocks) == 1
        assert "min_node_count" in blocks[0]

    def test_no_match(self):
        blocks = find_blocks(GOOD_GKE_CLUSTER, "nonexistent_block")
        assert len(blocks) == 0


class TestExtractValues:
    def test_string_values(self):
        vals = extract_string_values(GOOD_GKE_CLUSTER, "name")
        assert "ocr-local" in vals

    def test_number_values(self):
        vals = extract_number_values(GOOD_GKE_CLUSTER, "initial_node_count")
        assert 1 in vals

    def test_bool_values(self):
        vals = extract_bool_values(GOOD_GKE_CLUSTER, "remove_default_node_pool")
        assert True in vals

    def test_no_match(self):
        vals = extract_string_values(GOOD_GKE_CLUSTER, "nonexistent_key")
        assert len(vals) == 0


class TestHasBlock:
    def test_block_exists(self):
        assert has_block(GOOD_GKE_CLUSTER, "workload_identity_config")

    def test_block_missing(self):
        assert not has_block(GOOD_GKE_CLUSTER, "enable_autopilot")


class TestReadTfFiles:
    def test_reads_tf_files(self, good_tf_dir):
        files = read_tf_files(good_tf_dir)
        assert len(files) >= 1
        assert any("gke" in path for path in files)

    def test_empty_dir(self, empty_tf_dir):
        files = read_tf_files(empty_tf_dir)
        assert len(files) == 0

    def test_nonexistent_dir(self):
        files = read_tf_files("/nonexistent/path/does/not/exist")
        assert len(files) == 0


# ---------------------------------------------------------------------------
# Tests: Cluster Validation
# ---------------------------------------------------------------------------


class TestValidateGKECluster:
    def test_good_cluster_no_errors(self):
        tf_files = {"modules/gke/main.tf": GOOD_GKE_CLUSTER}
        findings = validate_gke_cluster(tf_files)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_workload_identity(self):
        content = GOOD_GKE_CLUSTER.replace(
            "workload_identity_config", "# workload_identity_disabled"
        )
        tf_files = {"main.tf": content}
        findings = validate_gke_cluster(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-010" in rule_ids

    def test_missing_vpc_native(self):
        content = GOOD_GKE_CLUSTER.replace(
            "ip_allocation_policy", "# ip_allocation_disabled"
        )
        tf_files = {"main.tf": content}
        findings = validate_gke_cluster(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-011" in rule_ids

    def test_missing_private_cluster(self):
        content = GOOD_GKE_CLUSTER.replace(
            "private_cluster_config", "# private_cluster_disabled"
        )
        tf_files = {"main.tf": content}
        findings = validate_gke_cluster(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-012" in rule_ids

    def test_strict_escalates_private_cluster(self):
        content = GOOD_GKE_CLUSTER.replace(
            "private_cluster_config", "# private_cluster_disabled"
        )
        tf_files = {"main.tf": content}
        findings = validate_gke_cluster(tf_files, strict=True)
        f012 = [f for f in findings if f.rule_id == "GKE-012"]
        assert any(f.severity == Severity.ERROR for f in f012)

    def test_missing_remove_default_pool(self):
        content = GOOD_GKE_CLUSTER.replace(
            "remove_default_node_pool = true",
            "remove_default_node_pool = false",
        )
        tf_files = {"main.tf": content}
        findings = validate_gke_cluster(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-014" in rule_ids

    def test_no_cluster_resource(self):
        tf_files = {"main.tf": "# empty terraform file\n"}
        findings = validate_gke_cluster(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-001" in rule_ids

    def test_rapid_channel_strict_warning(self):
        cluster_with_rapid = """
resource "google_container_cluster" "test" {
  name     = "test"
  project  = "test"
  location = "us-central1"

  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "test.svc.id.goog"
  }

  release_channel {
    channel = "RAPID"
  }

  resource_labels = var.labels
}
"""
        tf_files = {"main.tf": cluster_with_rapid}
        findings = validate_gke_cluster(tf_files, strict=True)
        f013 = [f for f in findings if f.rule_id == "GKE-013"]
        assert len(f013) >= 1


# ---------------------------------------------------------------------------
# Tests: Node Pool Validation
# ---------------------------------------------------------------------------


class TestValidateNodePools:
    def test_good_gpu_pool_no_errors(self):
        tf_files = {"main.tf": GOOD_GPU_NODE_POOL}
        findings = validate_node_pools(tf_files)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_incompatible_gpu_machine_type(self):
        content = GOOD_GPU_NODE_POOL.replace(
            'machine_type = "n1-standard-8"',
            'machine_type = "e2-standard-8"',
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-021" in rule_ids

    def test_missing_gpu_driver_config(self):
        content = GOOD_GPU_NODE_POOL.replace(
            "gpu_driver_installation_config", "# gpu_driver_disabled"
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-023" in rule_ids

    def test_small_gpu_disk(self):
        content = GOOD_GPU_NODE_POOL.replace(
            "disk_size_gb = 200", "disk_size_gb = 50"
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-024" in rule_ids

    def test_missing_gpu_taint(self):
        # Remove the taint block
        content = GOOD_GPU_NODE_POOL.replace("taint {", "# taint_removed {")
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-025" in rule_ids

    def test_invalid_autoscaling_range(self):
        content = GOOD_GPU_NODE_POOL.replace(
            "min_node_count = 0\n    max_node_count = 8",
            "min_node_count = 10\n    max_node_count = 5",
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        errors = [f for f in findings if f.rule_id == "GKE-026" and f.severity == Severity.ERROR]
        assert len(errors) >= 1

    def test_no_gpu_pool_warning(self):
        tf_files = {"main.tf": GOOD_CPU_NODE_POOL}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-020" in rule_ids

    def test_missing_autoscaling(self):
        content = GOOD_CPU_NODE_POOL.replace("autoscaling {", "# autoscaling_removed {")
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        f026 = [f for f in findings if f.rule_id == "GKE-026"]
        assert len(f026) >= 1

    def test_auto_repair_disabled(self):
        content = GOOD_GPU_NODE_POOL.replace(
            "auto_repair  = true", "auto_repair  = false"
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-028" in rule_ids

    def test_auto_upgrade_disabled(self):
        content = GOOD_GPU_NODE_POOL.replace(
            "auto_upgrade = true", "auto_upgrade = false"
        )
        tf_files = {"main.tf": content}
        findings = validate_node_pools(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-028" in rule_ids


# ---------------------------------------------------------------------------
# Tests: IAM Validation
# ---------------------------------------------------------------------------


class TestValidateIAMBindings:
    def test_good_iam_no_errors(self):
        tf_files = {"main.tf": GOOD_IAM}
        findings = validate_iam_bindings(tf_files)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_overly_permissive_role(self):
        content = """
resource "google_project_iam_member" "bad_binding" {
  project = "my-project"
  role    = "roles/editor"
  member  = "serviceAccount:test@test.iam.gserviceaccount.com"
}
"""
        tf_files = {"main.tf": content}
        findings = validate_iam_bindings(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-040" in rule_ids

    def test_iam_binding_warning(self):
        content = """
resource "google_project_iam_binding" "binding" {
  project = "my-project"
  role    = "roles/logging.logWriter"
  members = ["serviceAccount:test@test.iam.gserviceaccount.com"]
}
"""
        tf_files = {"main.tf": content}
        findings = validate_iam_bindings(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-041" in rule_ids

    def test_owner_role_blocked(self):
        content = """
resource "google_project_iam_member" "owner" {
  project = "my-project"
  role    = "roles/owner"
  member  = "serviceAccount:test@test.iam.gserviceaccount.com"
}
"""
        tf_files = {"main.tf": content}
        findings = validate_iam_bindings(tf_files)
        errors = [f for f in findings if f.rule_id == "GKE-040"]
        assert len(errors) >= 1
        assert errors[0].severity == Severity.ERROR


# ---------------------------------------------------------------------------
# Tests: Networking Validation
# ---------------------------------------------------------------------------


class TestValidateNetworking:
    def test_good_networking_no_errors(self):
        tf_files = {"main.tf": GOOD_NETWORKING}
        findings = validate_networking(tf_files)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_auto_create_subnetworks_warning(self):
        content = GOOD_NETWORKING.replace(
            "auto_create_subnetworks = false",
            "auto_create_subnetworks = true",
        )
        tf_files = {"main.tf": content}
        findings = validate_networking(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-050" in rule_ids

    def test_missing_secondary_ranges(self):
        content = """
resource "google_compute_subnetwork" "main" {
  name          = "test-subnet"
  project       = "test"
  region        = "us-central1"
  network       = "test-vpc"
  ip_cidr_range = "10.0.0.0/20"
  private_ip_google_access = true
}
"""
        tf_files = {"main.tf": content}
        findings = validate_networking(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-052" in rule_ids

    def test_permissive_firewall(self):
        content = GOOD_NETWORKING + """
resource "google_compute_firewall" "bad" {
  name    = "allow-all"
  network = "test"
  source_ranges = ["0.0.0.0/0"]
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
}
"""
        tf_files = {"main.tf": content}
        findings = validate_networking(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-053" in rule_ids

    def test_missing_nat_warning(self):
        content = """
resource "google_compute_network" "main" {
  name                    = "test"
  project                 = "test"
  auto_create_subnetworks = false
}
"""
        tf_files = {"main.tf": content}
        findings = validate_networking(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-054" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Labels Validation
# ---------------------------------------------------------------------------


class TestValidateLabels:
    def test_good_labels(self):
        tf_files = {"variables.tf": LABELS_VARIABLE}
        findings = validate_labels(tf_files)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert len(warnings) == 0

    def test_strict_recommends_extra_labels(self):
        tf_files = {"variables.tf": LABELS_VARIABLE}
        findings = validate_labels(tf_files, strict=True)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-061" in rule_ids

    def test_missing_required_label(self):
        content = """
variable "labels" {
  type = map(string)
  default = {
    project = "ocr-local"
  }
}
"""
        tf_files = {"variables.tf": content}
        findings = validate_labels(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-060" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Artifact Registry Validation
# ---------------------------------------------------------------------------


class TestValidateArtifactRegistry:
    def test_good_registry(self):
        tf_files = {"main.tf": GOOD_ARTIFACT_REGISTRY}
        findings = validate_artifact_registry(tf_files)
        assert len(findings) == 0

    def test_non_docker_format(self):
        content = GOOD_ARTIFACT_REGISTRY.replace(
            'format        = "DOCKER"',
            'format        = "MAVEN"',
        )
        tf_files = {"main.tf": content}
        findings = validate_artifact_registry(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-070" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Autopilot Detection
# ---------------------------------------------------------------------------


class TestValidateAutopilotDetection:
    def test_standard_mode_no_findings(self):
        tf_files = {"main.tf": GOOD_GKE_CLUSTER}
        findings = validate_autopilot_detection(tf_files)
        autopilot_findings = [f for f in findings if f.rule_id == "GKE-080"]
        assert len(autopilot_findings) == 0

    def test_autopilot_detected(self):
        content = """
resource "google_container_cluster" "autopilot" {
  name     = "autopilot-cluster"
  project  = "test"
  location = "us-central1"

  enable_autopilot = true
}
"""
        tf_files = {"main.tf": content}
        findings = validate_autopilot_detection(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-080" in rule_ids

    def test_autopilot_with_node_pools_warns(self):
        content = """
resource "google_container_cluster" "autopilot" {
  name     = "autopilot-cluster"
  project  = "test"
  location = "us-central1"

  enable_autopilot = true
}

resource "google_container_node_pool" "manual" {
  name     = "manual-pool"
  project  = "test"
  location = "us-central1"
  cluster  = "autopilot-cluster"
}
"""
        tf_files = {"main.tf": content}
        findings = validate_autopilot_detection(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-081" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Service Account Validation
# ---------------------------------------------------------------------------


class TestValidateServiceAccount:
    def test_sa_present(self):
        tf_files = {"main.tf": GOOD_IAM}
        findings = validate_service_account(tf_files)
        assert len(findings) == 0

    def test_missing_sa(self):
        tf_files = {"main.tf": "# no service account\n"}
        findings = validate_service_account(tf_files)
        rule_ids = {f.rule_id for f in findings}
        assert "GKE-090" in rule_ids


# ---------------------------------------------------------------------------
# Tests: Report Model
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_empty_report_passes(self):
        report = ValidationReport(terraform_dir="/test")
        assert report.passed
        assert report.error_count == 0

    def test_report_with_error_fails(self):
        report = ValidationReport(terraform_dir="/test")
        report.findings.append(Finding(
            rule_id="TEST-001",
            severity=Severity.ERROR,
            message="Test error",
        ))
        assert not report.passed
        assert report.error_count == 1

    def test_report_counts(self):
        report = ValidationReport(terraform_dir="/test")
        report.findings.extend([
            Finding(rule_id="E1", severity=Severity.ERROR, message="err"),
            Finding(rule_id="W1", severity=Severity.WARNING, message="warn"),
            Finding(rule_id="W2", severity=Severity.WARNING, message="warn2"),
            Finding(rule_id="I1", severity=Severity.INFO, message="info"),
        ])
        assert report.error_count == 1
        assert report.warning_count == 2
        assert report.info_count == 1

    def test_to_dict_structure(self):
        report = ValidationReport(terraform_dir="/test")
        report.findings.append(Finding(
            rule_id="T1",
            severity=Severity.WARNING,
            message="test",
        ))
        d = report.to_dict()
        assert "summary" in d
        assert "findings" in d
        assert d["summary"]["warnings"] == 1

    def test_to_markdown_contains_header(self):
        report = ValidationReport(terraform_dir="/test")
        md = report.to_markdown()
        assert "# GKE Terraform Validation Report" in md
        assert "PASSED" in md

    def test_to_markdown_with_findings(self):
        report = ValidationReport(terraform_dir="/test")
        report.findings.append(Finding(
            rule_id="GKE-010",
            severity=Severity.ERROR,
            message="Missing workload identity",
            recommendation="Add workload_identity_config",
        ))
        md = report.to_markdown()
        assert "FAILED" in md
        assert "GKE-010" in md
        assert "Recommendations" in md


class TestFinding:
    def test_to_dict(self):
        f = Finding(
            rule_id="GKE-010",
            severity=Severity.ERROR,
            message="test",
            file="main.tf",
            line=10,
        )
        d = f.to_dict()
        assert d["severity"] == "error"
        assert d["rule_id"] == "GKE-010"


# ---------------------------------------------------------------------------
# Tests: Full Validation
# ---------------------------------------------------------------------------


class TestValidateGKETerraform:
    def test_good_config_passes(self, good_tf_dir):
        report = validate_gke_terraform(good_tf_dir)
        assert report.error_count == 0

    def test_empty_dir_reports_error(self, empty_tf_dir):
        report = validate_gke_terraform(empty_tf_dir)
        assert report.error_count >= 1
        rule_ids = {f.rule_id for f in report.findings}
        assert "GKE-000" in rule_ids

    def test_nonexistent_dir(self):
        report = validate_gke_terraform("/nonexistent/terraform/path")
        assert report.error_count >= 1

    def test_actual_project_terraform(self):
        """Validate the actual project terraform directory if available."""
        project_tf = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "terraform",
        )
        if os.path.isdir(project_tf):
            report = validate_gke_terraform(project_tf)
            # Should have no errors (the project config follows best practices)
            assert report.error_count == 0


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_good_config(self, good_tf_dir):
        rc = main(["--terraform-dir", good_tf_dir])
        assert rc == 0

    def test_main_empty_dir(self, empty_tf_dir):
        rc = main(["--terraform-dir", empty_tf_dir])
        assert rc == 1

    def test_main_json_output(self, good_tf_dir, tmp_path):
        out_dir = str(tmp_path / "reports")
        rc = main(["--terraform-dir", good_tf_dir, "--output-dir", out_dir])
        assert rc == 0
        json_path = Path(out_dir) / "gke-validation-report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "findings" in data

    def test_main_markdown_output(self, good_tf_dir, tmp_path):
        out_dir = str(tmp_path / "reports")
        main(["--terraform-dir", good_tf_dir, "--output-dir", out_dir])
        md_path = Path(out_dir) / "gke-validation-report.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "GKE Terraform Validation Report" in content

    def test_main_strict_mode(self, good_tf_dir):
        rc = main(["--terraform-dir", good_tf_dir, "--strict"])
        # May or may not pass depending on strict escalations
        assert rc in (0, 1)

    def test_main_json_only(self, good_tf_dir, tmp_path, capsys):
        out_dir = str(tmp_path / "reports")
        rc = main([
            "--terraform-dir", good_tf_dir,
            "--output-dir", out_dir,
            "--json-only",
        ])
        assert rc == 0
        # Markdown report should not exist
        md_path = Path(out_dir) / "gke-validation-report.md"
        assert not md_path.exists()
        # JSON should still exist
        json_path = Path(out_dir) / "gke-validation-report.json"
        assert json_path.exists()
