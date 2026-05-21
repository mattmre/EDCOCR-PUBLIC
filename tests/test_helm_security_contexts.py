"""Tests for Helm chart container security contexts.

Validates that all services in values.yaml have hardened containerSecurityContext
with capabilities.drop=[ALL] and seccompProfile.type=RuntimeDefault per CIS
Kubernetes Benchmark recommendations.
"""

import os

import pytest
import yaml

VALUES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "helm", "ocr-local", "values.yaml"
)

# All service keys that have containerSecurityContext in values.yaml
ALL_SERVICE_KEYS = [
    "coordinator",
    "celeryCoordinator",
    "celeryBeat",
    "flower",
    "gpuWorker",
    "cpuOcrWorker",
    "layoutCpuWorker",
    "nlpGpuWorker",
    "layoutlmWorker",
    "cpuWorker",
    "postgresql",
    "rabbitmq",
    "redis",
]

# GPU workers that need SYS_PTRACE for NVIDIA profiling
GPU_WORKER_KEYS = ["gpuWorker", "nlpGpuWorker", "layoutlmWorker"]

# CPU-only services that must NOT have SYS_PTRACE
CPU_ONLY_KEYS = [k for k in ALL_SERVICE_KEYS if k not in GPU_WORKER_KEYS]


@pytest.fixture(scope="module")
def values():
    """Load the Helm values.yaml file."""
    with open(VALUES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("service_key", ALL_SERVICE_KEYS)
def test_service_has_container_security_context(values, service_key):
    """Each service must define a containerSecurityContext block."""
    assert service_key in values, f"Service key '{service_key}' missing from values.yaml"
    svc = values[service_key]
    assert "containerSecurityContext" in svc, (
        f"{service_key} missing containerSecurityContext"
    )


@pytest.mark.parametrize("service_key", ALL_SERVICE_KEYS)
def test_allow_privilege_escalation_false(values, service_key):
    """allowPrivilegeEscalation must be False for all services."""
    ctx = values[service_key]["containerSecurityContext"]
    assert ctx.get("allowPrivilegeEscalation") is False, (
        f"{service_key}: allowPrivilegeEscalation must be false"
    )


@pytest.mark.parametrize("service_key", ALL_SERVICE_KEYS)
def test_capabilities_drop_all(values, service_key):
    """capabilities.drop must contain 'ALL' for all services."""
    ctx = values[service_key]["containerSecurityContext"]
    caps = ctx.get("capabilities", {})
    drop = caps.get("drop", [])
    assert "ALL" in drop, (
        f"{service_key}: capabilities.drop must include 'ALL', got {drop}"
    )


@pytest.mark.parametrize("service_key", ALL_SERVICE_KEYS)
def test_seccomp_profile_runtime_default(values, service_key):
    """seccompProfile.type must be 'RuntimeDefault' for all services."""
    ctx = values[service_key]["containerSecurityContext"]
    profile = ctx.get("seccompProfile", {})
    assert profile.get("type") == "RuntimeDefault", (
        f"{service_key}: seccompProfile.type must be 'RuntimeDefault', "
        f"got {profile.get('type')}"
    )


@pytest.mark.parametrize("service_key", GPU_WORKER_KEYS)
def test_gpu_workers_have_sys_ptrace(values, service_key):
    """GPU workers must have capabilities.add containing SYS_PTRACE."""
    ctx = values[service_key]["containerSecurityContext"]
    caps = ctx.get("capabilities", {})
    add = caps.get("add", [])
    assert "SYS_PTRACE" in add, (
        f"{service_key}: GPU worker must have capabilities.add=['SYS_PTRACE'], got {add}"
    )


@pytest.mark.parametrize("service_key", CPU_ONLY_KEYS)
def test_cpu_services_no_extra_capabilities(values, service_key):
    """CPU-only services must not add any capabilities."""
    ctx = values[service_key]["containerSecurityContext"]
    caps = ctx.get("capabilities", {})
    add = caps.get("add", [])
    assert len(add) == 0, (
        f"{service_key}: CPU-only service should not add capabilities, got {add}"
    )


def test_all_known_services_covered(values):
    """Verify our service key list matches what is in values.yaml."""
    # Find all top-level keys that have containerSecurityContext
    found = [
        k for k, v in values.items()
        if isinstance(v, dict) and "containerSecurityContext" in v
    ]
    missing_from_list = set(found) - set(ALL_SERVICE_KEYS)
    missing_from_yaml = set(ALL_SERVICE_KEYS) - set(found)
    assert not missing_from_list, (
        f"Services in values.yaml not covered by test list: {missing_from_list}"
    )
    assert not missing_from_yaml, (
        f"Services in test list not found in values.yaml: {missing_from_yaml}"
    )


def test_total_container_security_context_count(values):
    """Verify the expected number of containerSecurityContext blocks."""
    count = sum(
        1 for v in values.values()
        if isinstance(v, dict) and "containerSecurityContext" in v
    )
    assert count == 13, (
        f"Expected 13 containerSecurityContext blocks, found {count}"
    )
