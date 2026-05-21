"""Tests for api.versioning — API stability contract enforcement."""

import unittest

from api.versioning import (
    API_SURFACE,
    CompatibilityReport,
    EndpointRecord,
    StabilityTier,
    check_backward_compatibility,
    get_api_version,
    get_beta_endpoints,
    get_experimental_endpoints,
    get_stable_endpoints,
    get_version_header,
)


class TestStabilityTier(unittest.TestCase):
    def test_tier_values(self):
        assert StabilityTier.STABLE.value == "stable"
        assert StabilityTier.BETA.value == "beta"
        assert StabilityTier.EXPERIMENTAL.value == "experimental"

    def test_all_tiers(self):
        assert len(StabilityTier) == 3


class TestEndpointRecord(unittest.TestCase):
    def test_creation(self):
        ep = EndpointRecord("GET", "/api/v1/test", "test_ep", StabilityTier.STABLE)
        assert ep.method == "GET"
        assert ep.path == "/api/v1/test"
        assert ep.name == "test_ep"
        assert ep.tier == StabilityTier.STABLE
        assert ep.auth_required is True
        assert ep.since_version == "1.0.0"
        assert ep.deprecated is False

    def test_frozen(self):
        ep = EndpointRecord("GET", "/test", "t", StabilityTier.STABLE)
        with self.assertRaises(AttributeError):
            ep.method = "POST"

    def test_deprecated_endpoint(self):
        ep = EndpointRecord("GET", "/old", "old", StabilityTier.STABLE, deprecated=True, deprecated_in="1.1.0")
        assert ep.deprecated is True
        assert ep.deprecated_in == "1.1.0"


class TestAPISurface(unittest.TestCase):
    def test_surface_is_tuple(self):
        assert isinstance(API_SURFACE, tuple)

    def test_surface_not_empty(self):
        assert len(API_SURFACE) >= 36

    def test_all_records_are_endpoint_records(self):
        for ep in API_SURFACE:
            assert isinstance(ep, EndpointRecord)

    def test_stable_endpoint_count(self):
        stable = get_stable_endpoints()
        assert len(stable) == 17  # 7 jobs + 5 batch + 5 health/readiness

    def test_beta_endpoint_count(self):
        beta = get_beta_endpoints()
        assert len(beta) == 30  # output contracts plus queue threshold endpoints

    def test_experimental_endpoint_count(self):
        exp = get_experimental_endpoints()
        assert len(exp) == 10  # 10 admin endpoints

    def test_health_no_auth(self):
        health = [
            e for e in API_SURFACE
            if e.name in {
                "health_check",
                "detailed_health_check",
                "ready_check",
                "readiness_check",
                "external_translation_readiness_check",
            }
        ]
        assert len(health) == 5
        assert all(e.auth_required is False for e in health)

    def test_vlm_health_no_auth(self):
        vlm = [e for e in API_SURFACE if e.name == "vlm_health"]
        assert len(vlm) == 1
        assert vlm[0].auth_required is False

    def test_all_paths_start_with_api_v1(self):
        for ep in API_SURFACE:
            assert ep.path.startswith("/api/v1/"), f"{ep.name}: {ep.path}"

    def test_unique_method_path_combinations(self):
        seen = set()
        for ep in API_SURFACE:
            key = (ep.method, ep.path)
            assert key not in seen, f"Duplicate: {ep.method} {ep.path}"
            seen.add(key)

    def test_valid_http_methods(self):
        valid = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        for ep in API_SURFACE:
            assert ep.method in valid, f"Invalid method: {ep.method}"


class TestGetAPIVersion(unittest.TestCase):
    def test_returns_string(self):
        v = get_api_version()
        assert isinstance(v, str)

    def test_semver_format(self):
        v = get_api_version()
        parts = v.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()


class TestGetVersionHeader(unittest.TestCase):
    def test_returns_dict(self):
        h = get_version_header()
        assert isinstance(h, dict)

    def test_has_version(self):
        h = get_version_header()
        assert "X-API-Version" in h

    def test_has_stability(self):
        h = get_version_header()
        assert h["X-API-Stability"] == "v1-stable"


class TestCompatibilityReport(unittest.TestCase):
    def test_defaults(self):
        r = CompatibilityReport()
        assert r.compatible is True
        assert r.removed_endpoints == []
        assert r.changed_methods == []
        assert r.new_endpoints == []
        assert r.warnings == []


class TestCheckBackwardCompatibility(unittest.TestCase):
    def test_identical_surfaces(self):
        report = check_backward_compatibility(API_SURFACE, API_SURFACE)
        assert report.compatible is True
        assert report.removed_endpoints == []

    def test_added_endpoint(self):
        extended = API_SURFACE + (
            EndpointRecord("GET", "/api/v1/new", "new_ep", StabilityTier.STABLE),
        )
        report = check_backward_compatibility(API_SURFACE, extended)
        assert report.compatible is True
        assert len(report.new_endpoints) == 1

    def test_removed_stable_endpoint(self):
        reduced = tuple(e for e in API_SURFACE if e.name != "submit_job")
        report = check_backward_compatibility(API_SURFACE, reduced)
        assert report.compatible is False
        assert len(report.changed_methods) >= 1 or len(report.removed_endpoints) >= 1

    def test_removed_beta_endpoint_ok(self):
        # Removing a beta endpoint should not break compatibility
        reduced = tuple(e for e in API_SURFACE if e.name != "list_transforms")
        report = check_backward_compatibility(API_SURFACE, reduced)
        assert report.compatible is True

    def test_auth_added_to_stable(self):
        modified = []
        for e in API_SURFACE:
            if e.name == "health_check":
                modified.append(EndpointRecord(
                    e.method, e.path, e.name, e.tier, auth_required=True,
                    since_version=e.since_version,
                ))
            else:
                modified.append(e)
        report = check_backward_compatibility(API_SURFACE, tuple(modified))
        assert report.compatible is False
        assert any("auth now required" in w for w in report.warnings)

    def test_empty_surfaces(self):
        report = check_backward_compatibility((), ())
        assert report.compatible is True

    def test_method_changed(self):
        modified = []
        for e in API_SURFACE:
            if e.name == "submit_job":
                modified.append(EndpointRecord(
                    "PUT", e.path, e.name, e.tier, e.auth_required, e.since_version,
                ))
            else:
                modified.append(e)
        report = check_backward_compatibility(API_SURFACE, tuple(modified))
        assert report.compatible is False


class TestFilterFunctions(unittest.TestCase):
    def test_stable_all_tier_stable(self):
        for ep in get_stable_endpoints():
            assert ep.tier == StabilityTier.STABLE

    def test_beta_all_tier_beta(self):
        for ep in get_beta_endpoints():
            assert ep.tier == StabilityTier.BETA

    def test_experimental_all_tier_experimental(self):
        for ep in get_experimental_endpoints():
            assert ep.tier == StabilityTier.EXPERIMENTAL

    def test_total_equals_surface(self):
        total = len(get_stable_endpoints()) + len(get_beta_endpoints()) + len(get_experimental_endpoints())
        assert total == len(API_SURFACE)


if __name__ == "__main__":
    unittest.main()
