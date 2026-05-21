"""Tests for the consolidated cost/SLA usage API (M-24).

Verifies that api.usage_consolidated correctly delegates to underlying
modules and produces the expected unified response shapes.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

from api.usage_consolidated import (
    CombinedUsageReport,
    PipelineCostBreakdown,
    PipelineUsageSummary,
    TenantCostSummary,
    TenantSlaSummary,
    get_combined_usage_report,
    get_tenant_cost_summary,
    get_tenant_sla_summary,
)

TENANT = "tenant_abc123"


# ---------------------------------------------------------------------------
# Fixtures: mock data from the standalone modules
# ---------------------------------------------------------------------------

_PIPELINE_COST_REPORT = {
    "tenant_id": TENANT,
    "usage": {
        "tenant_id": TENANT,
        "pages_processed": 100,
        "gpu_seconds": 45.5,
        "storage_bytes": 1024 * 1024 * 500,
        "api_calls": 20,
        "jobs_submitted": 5,
        "jobs_completed": 4,
        "jobs_failed": 1,
        "first_activity": "2026-03-01T00:00:00+00:00",
        "last_activity": "2026-03-24T12:00:00+00:00",
    },
    "cost": {
        "page_cost": 1.0,
        "gpu_cost": 0.0455,
        "storage_cost": 0.0238,
        "api_cost": 0.002,
        "total_cost": 1.0713,
        "currency": "USD",
    },
}


def _make_sla_report_dict():
    """Build a dict matching sla_monitoring.SLAReport after asdict()."""
    return {
        "tenant_id": TENANT,
        "report_time": "2026-03-24T12:00:00+00:00",
        "window_hours": 24,
        "slo_statuses": [
            {
                "definition": {
                    "name": "Availability",
                    "metric": "availability",
                    "target": 99.5,
                    "unit": "percent",
                    "comparison": "gte",
                },
                "current_value": 100.0,
                "compliant": True,
                "margin": 0.5,
                "window_start": "2026-03-23T12:00:00+00:00",
                "window_end": "2026-03-24T12:00:00+00:00",
                "sample_count": 10,
            }
        ],
        "overall_compliant": True,
        "violation_count": 0,
        "compliance_percentage": 100.0,
    }


_DB_COST_DATA = {
    "period": "2026-03",
    "jobs_submitted": 3,
    "pages_processed": 75,
    "storage_bytes_used": 1024 * 1024 * 200,
    "api_calls": 15,
    "processing_seconds": 120.0,
    "estimated_costs": {
        "currency": "USD",
        "page_cost_usd": 0.0,
        "storage_ingest_cost_usd": 0.0,
        "api_call_cost_usd": 0.0,
        "processing_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "storage_gib_ingested": 0.0,
        "processing_hours": 0.0,
        "rates": {
            "per_page_usd": 0.0,
            "per_gib_ingested_usd": 0.0,
            "per_api_call_usd": 0.0,
            "per_processing_hour_usd": 0.0,
        },
    },
}

_DB_SLO_DATA = {
    "tenant_id": TENANT,
    "window_hours": 24,
    "window_start": "2026-03-23T12:00:00",
    "window_end": "2026-03-24T12:00:00",
    "jobs_total": 5,
    "terminal_jobs": 4,
    "completed_jobs": 3,
    "failed_jobs": 1,
    "cancelled_jobs": 0,
    "active_jobs": 1,
    "pages_processed": 75,
    "success_rate": 0.75,
    "failure_rate": 0.25,
    "avg_processing_seconds": 30.0,
    "p95_processing_seconds": 50.0,
    "throughput_jobs_per_hour": 0.125,
    "throughput_pages_per_hour": 3.125,
    "targets": {
        "success_rate_min": 0.95,
        "p95_processing_seconds_max": 1800.0,
    },
    "status": {
        "success_rate_met": False,
        "p95_processing_met": True,
        "overall_met": False,
    },
}


# ---------------------------------------------------------------------------
# Helper: stub module for cost_tracking / sla_monitoring
# ---------------------------------------------------------------------------


def _stub_cost_tracking_module(report_data):
    """Create a fake cost_tracking module with a get_tracker() that returns
    a mock tracker whose get_cost_report returns *report_data*.
    """
    mod = ModuleType("cost_tracking")
    tracker = MagicMock()
    tracker.get_cost_report.return_value = report_data
    mod.get_tracker = MagicMock(return_value=tracker)
    return mod


def _stub_sla_monitoring_module(report_data):
    """Create a fake sla_monitoring module with a get_monitor() that returns
    a mock monitor whose evaluate_tenant returns a mock SLAReport.
    """
    mod = ModuleType("sla_monitoring")
    monitor = MagicMock()

    # evaluate_tenant should return an object that asdict() can handle.
    # We mock asdict at the call site instead -- just return the dict directly.
    report_obj = MagicMock()
    report_obj.__dataclass_fields__ = {}  # make asdict detect it
    monitor.evaluate_tenant.return_value = report_obj
    mod.get_monitor = MagicMock(return_value=monitor)
    return mod, report_obj


# ===================================================================
# Tests: get_tenant_cost_summary
# ===================================================================


class TestGetTenantCostSummary:
    """Tests for get_tenant_cost_summary delegation and response shape."""

    def test_pipeline_only(self):
        """When only the pipeline tracker has data, source is 'pipeline'."""
        mod = _stub_cost_tracking_module(_PIPELINE_COST_REPORT)
        with (
            patch.dict(sys.modules, {"cost_tracking": mod}),
            patch(
                "api.usage_consolidated._get_db_cost_data", return_value=None
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        assert isinstance(result, TenantCostSummary)
        assert result.tenant_id == TENANT
        assert result.source == "pipeline"
        assert result.pipeline_usage is not None
        assert result.pipeline_cost is not None
        assert result.db_cost is None
        assert result.pipeline_usage.pages_processed == 100
        assert result.pipeline_cost.total_cost == 1.0713

    def test_database_only(self):
        """When only the DB usage exists, source is 'database'."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=_DB_COST_DATA,
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        assert result.source == "database"
        assert result.pipeline_usage is None
        assert result.pipeline_cost is None
        assert result.db_cost is not None
        assert result.db_cost["pages_processed"] == 75

    def test_both_sources(self):
        """When both sources have data, source is 'both'."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=_DB_COST_DATA,
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        assert result.source == "both"
        assert result.pipeline_usage is not None
        assert result.db_cost is not None

    def test_no_sources(self):
        """When neither source has data, source is 'none'."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        assert result.source == "none"
        assert result.pipeline_usage is None
        assert result.pipeline_cost is None
        assert result.db_cost is None

    def test_pipeline_usage_fields(self):
        """Pipeline usage summary has all expected fields populated."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        usage = result.pipeline_usage
        assert isinstance(usage, PipelineUsageSummary)
        assert usage.tenant_id == TENANT
        assert usage.gpu_seconds == 45.5
        assert usage.storage_bytes == 1024 * 1024 * 500
        assert usage.api_calls == 20
        assert usage.jobs_submitted == 5
        assert usage.jobs_completed == 4
        assert usage.jobs_failed == 1

    def test_pipeline_cost_fields(self):
        """Pipeline cost breakdown has all expected fields populated."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ),
        ):
            result = get_tenant_cost_summary(TENANT)

        cost = result.pipeline_cost
        assert isinstance(cost, PipelineCostBreakdown)
        assert cost.page_cost == 1.0
        assert cost.gpu_cost == 0.0455
        assert cost.currency == "USD"

    def test_period_forwarded_to_db(self):
        """The optional period argument is forwarded to the DB query."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ) as mock_db,
        ):
            get_tenant_cost_summary(TENANT, period="2026-01")
            mock_db.assert_called_once_with(TENANT, period="2026-01")


# ===================================================================
# Tests: get_tenant_sla_summary
# ===================================================================


class TestGetTenantSlaSummary:
    """Tests for get_tenant_sla_summary delegation and response shape."""

    def test_pipeline_only(self):
        """When only the pipeline monitor has data, source is 'pipeline'."""
        sla_data = _make_sla_report_dict()
        with (
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=sla_data,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=None,
            ),
        ):
            result = get_tenant_sla_summary(TENANT)

        assert isinstance(result, TenantSlaSummary)
        assert result.source == "pipeline"
        assert result.pipeline_sla is not None
        assert result.db_slo is None
        assert result.pipeline_sla["overall_compliant"] is True

    def test_database_only(self):
        """When only the DB SLO snapshot exists, source is 'database'."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_tenant_sla_summary(TENANT)

        assert result.source == "database"
        assert result.pipeline_sla is None
        assert result.db_slo is not None
        assert result.db_slo["success_rate"] == 0.75

    def test_both_sources(self):
        """When both sources have data, source is 'both'."""
        sla_data = _make_sla_report_dict()
        with (
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=sla_data,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_tenant_sla_summary(TENANT)

        assert result.source == "both"

    def test_no_sources(self):
        """When neither source has data, source is 'none'."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=None,
            ),
        ):
            result = get_tenant_sla_summary(TENANT)

        assert result.source == "none"

    def test_window_hours_forwarded_to_db(self):
        """The optional window_hours argument is forwarded to the DB query."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=None,
            ) as mock_db,
        ):
            get_tenant_sla_summary(TENANT, window_hours=48)
            mock_db.assert_called_once_with(TENANT, window_hours=48)


# ===================================================================
# Tests: get_combined_usage_report
# ===================================================================


class TestGetCombinedUsageReport:
    """Tests for the combined report that merges all sources."""

    def test_combined_shape(self):
        """Combined report has the expected top-level structure."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=_DB_COST_DATA,
            ),
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=_make_sla_report_dict(),
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_combined_usage_report(TENANT)

        assert isinstance(result, CombinedUsageReport)
        assert result.tenant_id == TENANT
        assert result.report_time  # non-empty ISO timestamp
        assert isinstance(result.cost, TenantCostSummary)
        assert isinstance(result.sla, TenantSlaSummary)
        assert result.cost.source == "both"
        assert result.sla.source == "both"

    def test_sources_available_all(self):
        """When all four sources contribute, all are listed."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=_DB_COST_DATA,
            ),
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=_make_sla_report_dict(),
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_combined_usage_report(TENANT)

        assert "cost_tracking" in result.sources_available
        assert "api_usage_db" in result.sources_available
        assert "sla_monitoring" in result.sources_available
        assert "api_slo_db" in result.sources_available

    def test_sources_available_none(self):
        """When no sources contribute, sources_available is empty."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=None,
            ),
        ):
            result = get_combined_usage_report(TENANT)

        assert result.sources_available == []
        assert result.cost.source == "none"
        assert result.sla.source == "none"

    def test_sources_available_partial(self):
        """When only pipeline cost and DB SLO contribute, the list reflects that."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=None,
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_combined_usage_report(TENANT)

        assert "cost_tracking" in result.sources_available
        assert "api_usage_db" not in result.sources_available
        assert "sla_monitoring" not in result.sources_available
        assert "api_slo_db" in result.sources_available

    def test_period_and_window_forwarded(self):
        """Optional period and sla_window_hours are forwarded to delegates."""
        with (
            patch(
                "api.usage_consolidated.get_tenant_cost_summary"
            ) as mock_cost,
            patch(
                "api.usage_consolidated.get_tenant_sla_summary"
            ) as mock_sla,
        ):
            mock_cost.return_value = TenantCostSummary(tenant_id=TENANT)
            mock_sla.return_value = TenantSlaSummary(tenant_id=TENANT)

            get_combined_usage_report(
                TENANT, period="2026-01", sla_window_hours=48
            )

            mock_cost.assert_called_once_with(TENANT, period="2026-01")
            mock_sla.assert_called_once_with(TENANT, window_hours=48)

    def test_json_serializable(self):
        """The combined report can be serialized to JSON via Pydantic."""
        with (
            patch(
                "api.usage_consolidated._get_pipeline_cost_data",
                return_value=_PIPELINE_COST_REPORT,
            ),
            patch(
                "api.usage_consolidated._get_db_cost_data",
                return_value=_DB_COST_DATA,
            ),
            patch(
                "api.usage_consolidated._get_pipeline_sla_data",
                return_value=_make_sla_report_dict(),
            ),
            patch(
                "api.usage_consolidated._get_db_slo_data",
                return_value=_DB_SLO_DATA,
            ),
        ):
            result = get_combined_usage_report(TENANT)

        json_str = result.model_dump_json()
        assert TENANT in json_str
        assert "cost_tracking" in json_str


# ===================================================================
# Tests: delegation to real standalone modules
# ===================================================================


class TestPipelineCostDelegation:
    """Verify _get_pipeline_cost_data delegates to cost_tracking.get_tracker."""

    def test_delegates_to_cost_tracker(self):
        """The function calls get_tracker().get_cost_report(tenant_id)."""
        from api.usage_consolidated import _get_pipeline_cost_data

        mod = _stub_cost_tracking_module({"usage": {}, "cost": {}})
        with patch.dict(sys.modules, {"cost_tracking": mod}):
            result = _get_pipeline_cost_data(TENANT)

        mod.get_tracker.assert_called_once()
        mod.get_tracker().get_cost_report.assert_called_once_with(TENANT)
        assert result == {"usage": {}, "cost": {}}

    def test_returns_none_when_no_data(self):
        """Returns None when the tracker has no data for the tenant."""
        from api.usage_consolidated import _get_pipeline_cost_data

        mod = _stub_cost_tracking_module(None)
        with patch.dict(sys.modules, {"cost_tracking": mod}):
            result = _get_pipeline_cost_data(TENANT)

        assert result is None

    def test_returns_none_when_module_missing(self):
        """Returns None when cost_tracking module cannot be imported."""
        from api.usage_consolidated import _get_pipeline_cost_data

        # Temporarily remove cost_tracking if present
        with patch.dict(sys.modules, {"cost_tracking": None}):
            result = _get_pipeline_cost_data(TENANT)

        assert result is None


class TestPipelineSlaDelegation:
    """Verify _get_pipeline_sla_data delegates to sla_monitoring.get_monitor."""

    def test_delegates_to_sla_monitor(self):
        """The function calls get_monitor().evaluate_tenant(tenant_id)."""
        from api.usage_consolidated import _get_pipeline_sla_data

        expected = _make_sla_report_dict()
        # Patch asdict at the module level since the real code uses it
        with (
            patch(
                "api.usage_consolidated.asdict", return_value=expected
            ),
        ):
            mod, report_obj = _stub_sla_monitoring_module(expected)
            with patch.dict(sys.modules, {"sla_monitoring": mod}):
                result = _get_pipeline_sla_data(TENANT)

        mod.get_monitor.assert_called_once()
        mod.get_monitor().evaluate_tenant.assert_called_once_with(TENANT)
        assert result == expected

    def test_returns_none_when_module_missing(self):
        """Returns None when sla_monitoring module cannot be imported."""
        from api.usage_consolidated import _get_pipeline_sla_data

        with patch.dict(sys.modules, {"sla_monitoring": None}):
            result = _get_pipeline_sla_data(TENANT)

        assert result is None


# ===================================================================
# Tests: Pydantic model validation
# ===================================================================


class TestPydanticModels:
    """Verify Pydantic model defaults and serialization."""

    def test_pipeline_cost_breakdown_defaults(self):
        cost = PipelineCostBreakdown()
        assert cost.total_cost == 0.0
        assert cost.currency == "USD"

    def test_pipeline_usage_summary_required_field(self):
        usage = PipelineUsageSummary(tenant_id="t1")
        assert usage.pages_processed == 0
        assert usage.tenant_id == "t1"

    def test_tenant_cost_summary_defaults(self):
        s = TenantCostSummary(tenant_id="t1")
        assert s.source == "none"
        assert s.pipeline_usage is None
        assert s.pipeline_cost is None
        assert s.db_cost is None

    def test_tenant_sla_summary_defaults(self):
        s = TenantSlaSummary(tenant_id="t1")
        assert s.source == "none"
        assert s.pipeline_sla is None
        assert s.db_slo is None

    def test_combined_report_defaults(self):
        r = CombinedUsageReport(
            tenant_id="t1",
            report_time="2026-03-24T00:00:00+00:00",
            cost=TenantCostSummary(tenant_id="t1"),
            sla=TenantSlaSummary(tenant_id="t1"),
        )
        assert r.sources_available == []
        assert r.cost.source == "none"
        assert r.sla.source == "none"

    def test_model_dump_roundtrip(self):
        """Pydantic model_dump produces a dict that can re-create the model."""
        original = TenantCostSummary(
            tenant_id="t1",
            pipeline_usage=PipelineUsageSummary(tenant_id="t1", pages_processed=42),
            pipeline_cost=PipelineCostBreakdown(total_cost=1.5),
            source="pipeline",
        )
        data = original.model_dump()
        restored = TenantCostSummary(**data)
        assert restored.pipeline_usage.pages_processed == 42
        assert restored.pipeline_cost.total_cost == 1.5
        assert restored.source == "pipeline"
