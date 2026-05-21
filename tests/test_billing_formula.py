"""Tests for versioned billing formula and SLA targets.

Covers:
- BillingFormula dataclass construction and defaults
- Formula version locking
- Rate positivity validation
- JSON serialization / deserialization round-trip
- SLATargets dataclass construction and defaults
- SLA targets validation
- Effective date parsing
- Error cases (bad version, negative rates, unlocked)
"""

import json

import pytest

from cost_tracking import (
    BILLING_FORMULA_VERSION,
    COST_PER_API_CALL,
    COST_PER_GB_STORED,
    COST_PER_GPU_SECOND,
    COST_PER_PAGE,
    BillingFormula,
    get_billing_formula,
    validate_billing_formula,
)
from sla_monitoring import (
    DEFAULT_AVAILABILITY_TARGET,
    DEFAULT_ERROR_RATE_BUDGET,
    DEFAULT_P95_LATENCY_TARGET,
    DEFAULT_RECOVERY_TIME_TARGET,
    DEFAULT_THROUGHPUT_TARGET,
    SLA_FORMULA_VERSION,
    SLATargets,
    get_sla_targets,
    validate_sla_targets,
)

# ---------------------------------------------------------------------------
# BillingFormula: construction and defaults
# ---------------------------------------------------------------------------


class TestBillingFormulaDefaults:
    """Verify default values match existing module constants."""

    def test_version_constant(self):
        assert BILLING_FORMULA_VERSION == "1.0.0"

    def test_default_version(self):
        formula = BillingFormula()
        assert formula.version == BILLING_FORMULA_VERSION

    def test_default_cpu_cost(self):
        formula = BillingFormula()
        assert formula.cpu_cost_per_page == COST_PER_PAGE

    def test_default_gpu_cost(self):
        formula = BillingFormula()
        assert formula.gpu_cost_per_page == COST_PER_GPU_SECOND

    def test_default_storage_cost(self):
        formula = BillingFormula()
        assert formula.storage_cost_per_gb_month == COST_PER_GB_STORED

    def test_default_api_call_cost(self):
        formula = BillingFormula()
        assert formula.api_call_cost == COST_PER_API_CALL

    def test_default_locked(self):
        formula = BillingFormula()
        assert formula.locked is True

    def test_effective_date_is_today_format(self):
        formula = BillingFormula()
        # Should be YYYY-MM-DD
        parts = formula.effective_date.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2
        assert len(parts[2]) == 2


# ---------------------------------------------------------------------------
# BillingFormula: version locking
# ---------------------------------------------------------------------------


class TestBillingFormulaLocking:
    """Verify the locked billing formula cannot be accidentally unlocked."""

    def test_get_billing_formula_returns_locked(self):
        formula = get_billing_formula()
        assert formula.locked is True

    def test_get_billing_formula_version(self):
        formula = get_billing_formula()
        assert formula.version == BILLING_FORMULA_VERSION

    def test_locked_formula_validates_clean(self):
        formula = get_billing_formula()
        errors = validate_billing_formula(formula)
        assert errors == []

    def test_unlocked_formula_fails_validation(self):
        formula = BillingFormula(locked=False)
        errors = validate_billing_formula(formula)
        assert any("not locked" in e for e in errors)


# ---------------------------------------------------------------------------
# BillingFormula: rate validation
# ---------------------------------------------------------------------------


class TestBillingFormulaRateValidation:
    """All rates must be positive."""

    def test_all_default_rates_positive(self):
        formula = get_billing_formula()
        assert formula.cpu_cost_per_page > 0
        assert formula.gpu_cost_per_page > 0
        assert formula.storage_cost_per_gb_month > 0
        assert formula.api_call_cost > 0

    def test_zero_cpu_cost_fails(self):
        formula = BillingFormula(cpu_cost_per_page=0.0)
        errors = validate_billing_formula(formula)
        assert any("cpu_cost_per_page" in e for e in errors)

    def test_negative_gpu_cost_fails(self):
        formula = BillingFormula(gpu_cost_per_page=-0.01)
        errors = validate_billing_formula(formula)
        assert any("gpu_cost_per_page" in e for e in errors)

    def test_negative_storage_cost_fails(self):
        formula = BillingFormula(storage_cost_per_gb_month=-1.0)
        errors = validate_billing_formula(formula)
        assert any("storage_cost_per_gb_month" in e for e in errors)

    def test_zero_api_call_cost_fails(self):
        formula = BillingFormula(api_call_cost=0.0)
        errors = validate_billing_formula(formula)
        assert any("api_call_cost" in e for e in errors)

    def test_multiple_invalid_rates(self):
        formula = BillingFormula(
            cpu_cost_per_page=0.0,
            gpu_cost_per_page=-1.0,
            storage_cost_per_gb_month=0.0,
            api_call_cost=-0.5,
        )
        errors = validate_billing_formula(formula)
        assert len(errors) == 4  # one per rate field


# ---------------------------------------------------------------------------
# BillingFormula: version validation
# ---------------------------------------------------------------------------


class TestBillingFormulaVersionValidation:
    """Version must match semantic versioning format."""

    def test_valid_version(self):
        formula = BillingFormula(version="1.0.0")
        errors = validate_billing_formula(formula)
        assert not any("version" in e.lower() for e in errors)

    def test_valid_higher_version(self):
        formula = BillingFormula(version="2.1.3")
        errors = validate_billing_formula(formula)
        assert not any("version" in e.lower() for e in errors)

    def test_invalid_version_no_patch(self):
        formula = BillingFormula(version="1.0")
        errors = validate_billing_formula(formula)
        assert any("version" in e.lower() for e in errors)

    def test_invalid_version_alpha(self):
        formula = BillingFormula(version="abc")
        errors = validate_billing_formula(formula)
        assert any("version" in e.lower() for e in errors)

    def test_invalid_version_empty(self):
        formula = BillingFormula(version="")
        errors = validate_billing_formula(formula)
        assert any("version" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# BillingFormula: JSON serialization
# ---------------------------------------------------------------------------


class TestBillingFormulaSerialization:
    """Verify formula can be serialized to JSON and back."""

    def test_to_dict(self):
        formula = get_billing_formula()
        d = formula.to_dict()
        assert isinstance(d, dict)
        assert d["version"] == BILLING_FORMULA_VERSION
        assert d["locked"] is True

    def test_to_json(self):
        formula = get_billing_formula()
        j = formula.to_json()
        data = json.loads(j)
        assert data["version"] == BILLING_FORMULA_VERSION
        assert data["cpu_cost_per_page"] == COST_PER_PAGE
        assert data["gpu_cost_per_page"] == COST_PER_GPU_SECOND
        assert data["storage_cost_per_gb_month"] == COST_PER_GB_STORED
        assert data["api_call_cost"] == COST_PER_API_CALL
        assert data["locked"] is True

    def test_json_round_trip(self):
        formula = get_billing_formula()
        j = formula.to_json()
        data = json.loads(j)
        restored = BillingFormula(**data)
        assert restored.version == formula.version
        assert restored.cpu_cost_per_page == formula.cpu_cost_per_page
        assert restored.gpu_cost_per_page == formula.gpu_cost_per_page
        assert restored.storage_cost_per_gb_month == formula.storage_cost_per_gb_month
        assert restored.api_call_cost == formula.api_call_cost
        assert restored.locked == formula.locked
        assert restored.effective_date == formula.effective_date

    def test_to_dict_contains_all_fields(self):
        formula = get_billing_formula()
        d = formula.to_dict()
        expected_keys = {
            "version",
            "cpu_cost_per_page",
            "gpu_cost_per_page",
            "storage_cost_per_gb_month",
            "api_call_cost",
            "effective_date",
            "locked",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# BillingFormula: effective date validation
# ---------------------------------------------------------------------------


class TestBillingFormulaEffectiveDate:
    """Effective date must be a valid ISO date."""

    def test_valid_date(self):
        formula = BillingFormula(effective_date="2026-03-15")
        errors = validate_billing_formula(formula)
        assert not any("effective_date" in e for e in errors)

    def test_invalid_date_format(self):
        formula = BillingFormula(effective_date="03/15/2026")
        errors = validate_billing_formula(formula)
        assert any("effective_date" in e for e in errors)

    def test_invalid_date_garbage(self):
        formula = BillingFormula(effective_date="not-a-date")
        errors = validate_billing_formula(formula)
        assert any("effective_date" in e for e in errors)

    def test_empty_date(self):
        formula = BillingFormula(effective_date="")
        errors = validate_billing_formula(formula)
        assert any("effective_date" in e for e in errors)

    def test_date_with_time_fails(self):
        formula = BillingFormula(effective_date="2026-03-15T12:00:00")
        errors = validate_billing_formula(formula)
        assert any("effective_date" in e for e in errors)


# ---------------------------------------------------------------------------
# SLATargets: construction and defaults
# ---------------------------------------------------------------------------


class TestSLATargetsDefaults:
    """Verify SLA target defaults match existing module constants."""

    def test_version_constant(self):
        assert SLA_FORMULA_VERSION == "1.0.0"

    def test_default_version(self):
        targets = SLATargets()
        assert targets.version == SLA_FORMULA_VERSION

    def test_default_uptime(self):
        targets = SLATargets()
        # DEFAULT_AVAILABILITY_TARGET is 99.5 (percent), stored as 0.995
        assert targets.uptime_target == pytest.approx(
            DEFAULT_AVAILABILITY_TARGET / 100.0
        )

    def test_default_p95_latency_ms(self):
        targets = SLATargets()
        # DEFAULT_P95_LATENCY_TARGET is 30.0 seconds, stored as 30000.0 ms
        assert targets.p95_latency_ms == pytest.approx(
            DEFAULT_P95_LATENCY_TARGET * 1000.0
        )

    def test_default_error_rate(self):
        targets = SLATargets()
        # DEFAULT_ERROR_RATE_BUDGET is 1.0 (percent), stored as 0.01
        assert targets.error_rate_target == pytest.approx(
            DEFAULT_ERROR_RATE_BUDGET / 100.0
        )

    def test_default_throughput(self):
        targets = SLATargets()
        assert targets.throughput_ppm_target == DEFAULT_THROUGHPUT_TARGET

    def test_default_recovery_time(self):
        targets = SLATargets()
        assert targets.recovery_time_seconds == DEFAULT_RECOVERY_TIME_TARGET


# ---------------------------------------------------------------------------
# SLATargets: get and validate
# ---------------------------------------------------------------------------


class TestSLATargetsGetAndValidate:
    """Test get_sla_targets and validate_sla_targets functions."""

    def test_get_sla_targets_returns_instance(self):
        targets = get_sla_targets()
        assert isinstance(targets, SLATargets)

    def test_get_sla_targets_version(self):
        targets = get_sla_targets()
        assert targets.version == SLA_FORMULA_VERSION

    def test_default_targets_validate_clean(self):
        targets = get_sla_targets()
        errors = validate_sla_targets(targets)
        assert errors == []

    def test_invalid_version(self):
        targets = SLATargets(version="bad")
        errors = validate_sla_targets(targets)
        assert any("version" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# SLATargets: field validation
# ---------------------------------------------------------------------------


class TestSLATargetsFieldValidation:
    """Validate individual SLA target constraints."""

    def test_uptime_zero_fails(self):
        targets = SLATargets(uptime_target=0.0)
        errors = validate_sla_targets(targets)
        assert any("uptime_target" in e for e in errors)

    def test_uptime_negative_fails(self):
        targets = SLATargets(uptime_target=-0.5)
        errors = validate_sla_targets(targets)
        assert any("uptime_target" in e for e in errors)

    def test_uptime_above_one_fails(self):
        targets = SLATargets(uptime_target=1.5)
        errors = validate_sla_targets(targets)
        assert any("uptime_target" in e for e in errors)

    def test_uptime_exactly_one_passes(self):
        targets = SLATargets(uptime_target=1.0)
        errors = validate_sla_targets(targets)
        assert not any("uptime_target" in e for e in errors)

    def test_negative_latency_fails(self):
        targets = SLATargets(p95_latency_ms=-100.0)
        errors = validate_sla_targets(targets)
        assert any("p95_latency_ms" in e for e in errors)

    def test_zero_latency_fails(self):
        targets = SLATargets(p95_latency_ms=0.0)
        errors = validate_sla_targets(targets)
        assert any("p95_latency_ms" in e for e in errors)

    def test_error_rate_negative_fails(self):
        targets = SLATargets(error_rate_target=-0.01)
        errors = validate_sla_targets(targets)
        assert any("error_rate_target" in e for e in errors)

    def test_error_rate_one_fails(self):
        targets = SLATargets(error_rate_target=1.0)
        errors = validate_sla_targets(targets)
        assert any("error_rate_target" in e for e in errors)

    def test_error_rate_zero_passes(self):
        targets = SLATargets(error_rate_target=0.0)
        errors = validate_sla_targets(targets)
        assert not any("error_rate_target" in e for e in errors)

    def test_negative_throughput_fails(self):
        targets = SLATargets(throughput_ppm_target=-5.0)
        errors = validate_sla_targets(targets)
        assert any("throughput_ppm_target" in e for e in errors)

    def test_zero_throughput_fails(self):
        targets = SLATargets(throughput_ppm_target=0.0)
        errors = validate_sla_targets(targets)
        assert any("throughput_ppm_target" in e for e in errors)

    def test_negative_recovery_time_fails(self):
        targets = SLATargets(recovery_time_seconds=-1.0)
        errors = validate_sla_targets(targets)
        assert any("recovery_time_seconds" in e for e in errors)

    def test_zero_recovery_time_fails(self):
        targets = SLATargets(recovery_time_seconds=0.0)
        errors = validate_sla_targets(targets)
        assert any("recovery_time_seconds" in e for e in errors)


# ---------------------------------------------------------------------------
# SLATargets: serialization
# ---------------------------------------------------------------------------


class TestSLATargetsSerialization:
    """Verify SLA targets can be serialized to JSON and back."""

    def test_to_dict(self):
        targets = get_sla_targets()
        d = targets.to_dict()
        assert isinstance(d, dict)
        assert d["version"] == SLA_FORMULA_VERSION

    def test_to_json(self):
        targets = get_sla_targets()
        j = targets.to_json()
        data = json.loads(j)
        assert data["version"] == SLA_FORMULA_VERSION
        assert data["uptime_target"] == pytest.approx(
            DEFAULT_AVAILABILITY_TARGET / 100.0
        )
        assert data["throughput_ppm_target"] == DEFAULT_THROUGHPUT_TARGET

    def test_json_round_trip(self):
        targets = get_sla_targets()
        j = targets.to_json()
        data = json.loads(j)
        restored = SLATargets(**data)
        assert restored.version == targets.version
        assert restored.uptime_target == targets.uptime_target
        assert restored.p95_latency_ms == targets.p95_latency_ms
        assert restored.error_rate_target == targets.error_rate_target
        assert restored.throughput_ppm_target == targets.throughput_ppm_target
        assert restored.recovery_time_seconds == targets.recovery_time_seconds

    def test_to_dict_contains_all_fields(self):
        targets = get_sla_targets()
        d = targets.to_dict()
        expected_keys = {
            "version",
            "uptime_target",
            "p95_latency_ms",
            "error_rate_target",
            "throughput_ppm_target",
            "recovery_time_seconds",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    """Verify billing formula and SLA targets use the same version scheme."""

    def test_same_version_format(self):
        billing = get_billing_formula()
        sla = get_sla_targets()
        assert billing.version == sla.version

    def test_both_validate_clean(self):
        billing_errors = validate_billing_formula(get_billing_formula())
        sla_errors = validate_sla_targets(get_sla_targets())
        assert billing_errors == []
        assert sla_errors == []
