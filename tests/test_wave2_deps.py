"""Consumer-visible installer constraints, not requirement spelling checks."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _requirements(relative):
    requirements = {}
    for raw in (ROOT / relative).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            requirements[requirement.name.lower().replace("_", "-")] = requirement
    return requirements


@pytest.mark.parametrize(
    ("file", "package", "accepted", "rejected"),
    [
        ("requirements.txt", "paddlepaddle", "3.0.0", ("2.9.0", "3.0.1")),
        ("requirements.txt", "protobuf", "7.35.1", ("7.35.0", "7.35.2")),
        ("requirements.txt", "opencv-python-headless", "4.11.0.86", ("4.11.0.85", "4.12.0.88")),
        ("requirements.txt", "uvicorn", "0.54.0", ("1.0.0",)),
        ("requirements.txt", "prometheus-client", "0.26.0", ("1.0.0",)),
        ("requirements.txt", "hypothesis", "6.168.5", ("6.156.1",)),
        ("requirements.txt", "watchdog", "6.0.0", ("5.0.0",)),
        ("coordinator/requirements.txt", "redis", "8.1.0", ("8.0.0", "9.0.0")),
        ("coordinator/requirements.txt", "boto3", "1.43.110", ("1.43.41", "2.0.0")),
        ("coordinator/requirements.txt", "dnspython", "2.9.0", ("2.7.0", "3.0.0")),
        ("coordinator/requirements.txt", "prometheus-client", "0.26.0", ("1.0.0",)),
        ("coordinator/requirements.txt", "celery", "5.6.3", ("5.6.2", "5.6.4")),
        ("requirements-translation.txt", "ctranslate2", "4.8.2", ("4.8.0", "5.0.0")),
    ],
)
def test_installer_admits_qualified_version_and_rejects_unsupported_versions(
    file, package, accepted, rejected
):
    constraint = _requirements(file)[package].specifier
    assert accepted in constraint
    for version in rejected:
        assert version not in constraint


@pytest.mark.parametrize(
    ("platform", "python", "admitted"),
    [("Linux", "3.11", True), ("Linux", "3.12", False), ("Windows", "3.11", False)],
)
def test_legacy_uie_install_is_platform_and_python_gated(platform, python, admitted):
    requirement = _requirements("requirements-uie.txt")["paddlenlp"]
    assert requirement.marker.evaluate(
        {"platform_system": platform, "python_version": python}
    ) is admitted


def test_default_runtime_does_not_enable_optional_uie_or_duplicate_onnx_runtime():
    root = _requirements("requirements.txt")
    assert {"paddlenlp", "onnxruntime", "redis", "boto3"}.isdisjoint(root)
    assert not _requirements("coordinator/requirements.txt")["celery"].extras


def test_shared_prometheus_installer_constraints_agree():
    root = _requirements("requirements.txt")["prometheus-client"].specifier
    coordinator = _requirements("coordinator/requirements.txt")["prometheus-client"].specifier
    assert root == coordinator
