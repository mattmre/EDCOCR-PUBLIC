"""Tests for Wave 2 dependency version constraints.

Verifies that requirements files contain the expected version range pins
for redis, uvicorn, boto3, and prometheus-client after the Wave 2 bump.
"""

import pathlib
import re

import pytest
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_REQUIREMENTS = ROOT / "requirements.txt"
COORDINATOR_REQUIREMENTS = ROOT / "coordinator" / "requirements.txt"
UIE_REQUIREMENTS = ROOT / "requirements-uie.txt"
TRANSLATION_REQUIREMENTS = ROOT / "requirements-translation.txt"


def _parse_requirements(path: pathlib.Path) -> dict[str, str]:
    """Parse a requirements file into {normalised_name: raw_spec} mapping."""
    specs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip extras like [standard] or [rabbitmq] for name normalisation
        match = re.match(r"^([A-Za-z0-9_.-]+)(\[[^\]]*\])?(.*)", line)
        if match:
            name = match.group(1).lower().replace("-", "_")
            extras = match.group(2) or ""
            version_spec = match.group(3)
            specs[name] = f"{match.group(1)}{extras}{version_spec}"
    return specs


def _has_lower_bound_at_least(raw: str, minimum: str) -> bool:
    """Return True when a requirement contains a >= bound at or above minimum."""
    for op, version in re.findall(
        r"(>=)\s*([0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9_.-]*)?)", raw
    ):
        if op == ">=" and Version(version) >= Version(minimum):
            return True
    return False


# ---------------------------------------------------------------------------
# Root requirements.txt
# ---------------------------------------------------------------------------


class TestRootRequirements:
    """Validate Wave 2 pins in root requirements.txt."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.specs = _parse_requirements(ROOT_REQUIREMENTS)

    def test_uvicorn_lower_bound(self):
        """uvicorn must have a lower bound >= 0.48.0."""
        raw = self.specs.get("uvicorn", "")
        assert ">=0.48.0" in raw, f"uvicorn missing lower bound: {raw}"

    def test_uvicorn_upper_bound(self):
        """uvicorn must stay below 1.0.0."""
        raw = self.specs.get("uvicorn", "")
        assert "<1.0.0" in raw, f"uvicorn missing upper bound: {raw}"

    def test_prometheus_client_lower_bound(self):
        """prometheus-client must have a lower bound >= 0.25.0."""
        raw = self.specs.get("prometheus_client", "")
        assert ">=0.25.0" in raw, f"prometheus-client missing lower bound: {raw}"

    def test_prometheus_client_upper_bound(self):
        """prometheus-client must stay below 1.0.0."""
        raw = self.specs.get("prometheus_client", "")
        assert "<1.0.0" in raw, f"prometheus-client missing upper bound: {raw}"

    def test_no_redis_in_root(self):
        """redis should NOT appear in root requirements (coordinator-only)."""
        assert (
            "redis" not in self.specs
        ), "redis should only be in coordinator/requirements.txt"

    def test_no_boto3_in_root(self):
        """boto3 should NOT appear in root requirements (coordinator-only)."""
        assert (
            "boto3" not in self.specs
        ), "boto3 should only be in coordinator/requirements.txt"

    def test_opencv_pin_stays_on_numpy1_compatible_line(self):
        """OpenCV must stay on the highest Python 3.11 + numpy<2 compatible 4.x line."""
        raw = self.specs.get("opencv_python_headless", "")
        assert raw == "opencv-python-headless==4.11.0.86", (
            "opencv-python-headless pin drifted off the approved numpy<2-compatible "
            f"line: {raw}"
        )

    def test_paddlenlp_not_in_default_root_requirements(self):
        """PaddleNLP UIE must stay out of the default install path."""
        assert (
            "paddlenlp" not in self.specs
        ), "PaddleNLP belongs in requirements-uie.txt, not default requirements.txt"

    def test_optional_uie_requirements_carries_legacy_marker(self):
        """Legacy UIE remains explicit and excluded from Windows/Python 3.12."""
        uie_specs = _parse_requirements(UIE_REQUIREMENTS)
        raw = uie_specs.get("paddlenlp", "")
        assert raw == (
            'paddlenlp==2.8.1; platform_system != "Windows" '
            'and python_version < "3.12"'
        )

    def test_default_root_retains_patched_onnx_without_runtime_duplication(self):
        """Paddle 3 permits the existing patched ONNX dependency to remain."""
        assert self.specs.get("onnx") == "onnx>=1.21.0"
        assert "onnxruntime" not in self.specs

    def test_paddle_runtime_and_protobuf_pins(self):
        """The real-CPU-qualified Paddle and protobuf versions remain exact."""
        assert self.specs.get("paddlepaddle") == "paddlepaddle==3.0.0"
        assert self.specs.get("protobuf") == "protobuf==7.35.1"

    def test_hypothesis_lower_bound(self):
        """Hypothesis must include the reviewed property-test lower bound."""
        raw = self.specs.get("hypothesis", "")
        assert _has_lower_bound_at_least(
            raw, "6.156.2"
        ), f"hypothesis lower bound must be >= 6.156.2: {raw}"

    def test_watchdog_lower_bound(self):
        """Watchdog must include the reviewed hot-folder watcher lower bound."""
        raw = self.specs.get("watchdog", "")
        assert _has_lower_bound_at_least(
            raw, "6.0.0"
        ), f"watchdog lower bound must be >= 6.0.0: {raw}"


# ---------------------------------------------------------------------------
# Coordinator requirements.txt
# ---------------------------------------------------------------------------


class TestCoordinatorRequirements:
    """Validate Wave 2 pins in coordinator/requirements.txt."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.specs = _parse_requirements(COORDINATOR_REQUIREMENTS)

    def test_redis_lower_bound(self):
        """redis must have the qualified 8.0.1 lower bound."""
        raw = self.specs.get("redis", "")
        assert _has_lower_bound_at_least(
            raw, "8.0.1"
        ), f"redis lower bound must be >= 8.0.1: {raw}"

    def test_redis_upper_bound_blocks_9(self):
        """redis must stay below the next major line after qualification."""
        raw = self.specs.get("redis", "")
        assert "<9.0.0" in raw, f"redis missing upper bound: {raw}"

    def test_celery_rabbitmq_extra_removed(self):
        """Coordinator Celery pin must not use the nonexistent rabbitmq extra."""
        raw = self.specs.get("celery", "")
        assert raw == "celery==5.6.3", (
            "coordinator requirements must not use the ineffective "
            f"celery[rabbitmq] extra: {raw}"
        )

    def test_boto3_lower_bound(self):
        """boto3 must have a lower bound >= 1.43.42."""
        raw = self.specs.get("boto3", "")
        assert _has_lower_bound_at_least(
            raw, "1.43.42"
        ), f"boto3 lower bound must be >= 1.43.42: {raw}"

    def test_boto3_upper_bound(self):
        """boto3 must stay below 2.0.0."""
        raw = self.specs.get("boto3", "")
        assert "<2.0.0" in raw, f"boto3 missing upper bound: {raw}"

    def test_dnspython_lower_bound(self):
        """dnspython must have a lower bound >= 2.8.0."""
        raw = self.specs.get("dnspython", "")
        assert _has_lower_bound_at_least(
            raw, "2.8.0"
        ), f"dnspython lower bound must be >= 2.8.0: {raw}"

    def test_dnspython_upper_bound(self):
        """dnspython must stay below 3.0."""
        raw = self.specs.get("dnspython", "")
        assert "<3.0" in raw, f"dnspython missing upper bound: {raw}"

    def test_prometheus_client_lower_bound(self):
        """prometheus-client must have a lower bound >= 0.25.0."""
        raw = self.specs.get("prometheus_client", "")
        assert ">=0.25.0" in raw, f"prometheus-client missing lower bound: {raw}"

    def test_prometheus_client_upper_bound(self):
        """prometheus-client must stay below 1.0.0."""
        raw = self.specs.get("prometheus_client", "")
        assert "<1.0.0" in raw, f"prometheus-client missing upper bound: {raw}"


# ---------------------------------------------------------------------------
# Optional translation requirements.txt
# ---------------------------------------------------------------------------


class TestTranslationRequirements:
    """Validate optional translation runtime dependency bounds."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.specs = _parse_requirements(TRANSLATION_REQUIREMENTS)

    def test_ctranslate2_lower_bound(self):
        """CTranslate2 must include the v4.8.1 security/runtime fixes."""
        raw = self.specs.get("ctranslate2", "")
        assert _has_lower_bound_at_least(
            raw, "4.8.1"
        ), f"ctranslate2 lower bound must be >= 4.8.1: {raw}"

    def test_ctranslate2_upper_bound_blocks_5(self):
        """CTranslate2 major-version 5 remains a separate compatibility lane."""
        raw = self.specs.get("ctranslate2", "")
        assert "<5" in raw, f"ctranslate2 missing upper bound: {raw}"


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestCrossFileConsistency:
    """Verify that shared packages have consistent version specs across files."""

    def test_prometheus_client_consistent(self):
        """prometheus-client spec must match between root and coordinator."""
        root_specs = _parse_requirements(ROOT_REQUIREMENTS)
        coord_specs = _parse_requirements(COORDINATOR_REQUIREMENTS)

        root_raw = root_specs.get("prometheus_client", "")
        coord_raw = coord_specs.get("prometheus_client", "")

        # Extract just the version portion (after package name)
        root_ver = re.sub(r"^[A-Za-z0-9_.-]+(\[[^\]]*\])?", "", root_raw)
        coord_ver = re.sub(r"^[A-Za-z0-9_.-]+(\[[^\]]*\])?", "", coord_raw)

        assert root_ver == coord_ver, (
            f"prometheus-client version mismatch: "
            f"root={root_ver!r}, coordinator={coord_ver!r}"
        )

    def test_version_range_format(self):
        """All Wave 2 packages must use >=min,<max format (not ==)."""
        all_specs = {}
        all_specs.update(_parse_requirements(ROOT_REQUIREMENTS))
        all_specs.update(_parse_requirements(COORDINATOR_REQUIREMENTS))

        wave2_packages = ["uvicorn", "redis", "boto3", "prometheus_client"]
        for pkg in wave2_packages:
            raw = all_specs.get(pkg, "")
            if not raw:
                continue  # package not in this file, skip
            assert "==" not in raw, f"{pkg} still uses exact pin (==): {raw}"
            assert ">=" in raw, f"{pkg} missing >= lower bound: {raw}"
            assert "<" in raw, f"{pkg} missing < upper bound: {raw}"
