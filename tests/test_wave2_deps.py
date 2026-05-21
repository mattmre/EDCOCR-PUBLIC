"""Tests for Wave 2 dependency version constraints.

Verifies that requirements files contain the expected version range pins
for redis, uvicorn, boto3, and prometheus-client after the Wave 2 bump.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT_REQUIREMENTS = ROOT / "requirements.txt"
COORDINATOR_REQUIREMENTS = ROOT / "coordinator" / "requirements.txt"


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


# ---------------------------------------------------------------------------
# Root requirements.txt
# ---------------------------------------------------------------------------


class TestRootRequirements:
    """Validate Wave 2 pins in root requirements.txt."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.specs = _parse_requirements(ROOT_REQUIREMENTS)

    def test_uvicorn_lower_bound(self):
        """uvicorn must have a lower bound >= 0.42.0."""
        raw = self.specs.get("uvicorn", "")
        assert ">=0.42.0" in raw, f"uvicorn missing lower bound: {raw}"

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
        assert "redis" not in self.specs, (
            "redis should only be in coordinator/requirements.txt"
        )

    def test_no_boto3_in_root(self):
        """boto3 should NOT appear in root requirements (coordinator-only)."""
        assert "boto3" not in self.specs, (
            "boto3 should only be in coordinator/requirements.txt"
        )

    def test_opencv_pin_stays_on_numpy1_compatible_line(self):
        """OpenCV must stay on the highest Python 3.11 + numpy<2 compatible 4.x line."""
        raw = self.specs.get("opencv_python_headless", "")
        assert raw == "opencv-python-headless==4.11.0.86", (
            "opencv-python-headless pin drifted off the approved numpy<2-compatible "
            f"line: {raw}"
        )


# ---------------------------------------------------------------------------
# Coordinator requirements.txt
# ---------------------------------------------------------------------------


class TestCoordinatorRequirements:
    """Validate Wave 2 pins in coordinator/requirements.txt."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.specs = _parse_requirements(COORDINATOR_REQUIREMENTS)

    def test_redis_lower_bound(self):
        """redis must have a lower bound >= 7.0.1."""
        raw = self.specs.get("redis", "")
        assert ">=7.0.1" in raw, f"redis missing lower bound: {raw}"

    def test_redis_upper_bound_blocks_8(self):
        """redis must stay below 8.0.0 until a future major review is done."""
        raw = self.specs.get("redis", "")
        assert "<8.0.0" in raw, f"redis missing upper bound: {raw}"

    def test_boto3_lower_bound(self):
        """boto3 must have a lower bound >= 1.43.10."""
        raw = self.specs.get("boto3", "")
        assert ">=1.43.10" in raw, f"boto3 missing lower bound: {raw}"

    def test_boto3_upper_bound(self):
        """boto3 must stay below 2.0.0."""
        raw = self.specs.get("boto3", "")
        assert "<2.0.0" in raw, f"boto3 missing upper bound: {raw}"

    def test_prometheus_client_lower_bound(self):
        """prometheus-client must have a lower bound >= 0.25.0."""
        raw = self.specs.get("prometheus_client", "")
        assert ">=0.25.0" in raw, f"prometheus-client missing lower bound: {raw}"

    def test_prometheus_client_upper_bound(self):
        """prometheus-client must stay below 1.0.0."""
        raw = self.specs.get("prometheus_client", "")
        assert "<1.0.0" in raw, f"prometheus-client missing upper bound: {raw}"


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
            assert "==" not in raw, (
                f"{pkg} still uses exact pin (==): {raw}"
            )
            assert ">=" in raw, f"{pkg} missing >= lower bound: {raw}"
            assert "<" in raw, f"{pkg} missing < upper bound: {raw}"
