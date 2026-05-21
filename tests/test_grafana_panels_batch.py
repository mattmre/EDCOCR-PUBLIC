"""Tests for M-02/M-03/M-05/M-08: Grafana dashboard panels batch.

Validates:
- M-02: Cost-per-tenant stat panel exists with correct type and metric
- M-03: Per-tenant latency timeseries panel with p50/p95 histogram queries
- M-05: Queue depth per-stage panel with extraction/ocr/compression/nlp targets
- M-08: Template variables for $tenant_id (query) and $interval (6h/24h added)
- All new panel IDs are unique
- Total panel count is 47

Run with:
  python -m pytest tests/test_grafana_panels_batch.py -v
"""

from __future__ import annotations

import json
import os
import re

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HELM_TEMPLATES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "helm", "ocr-local", "templates",
)

_GRAFANA_DASHBOARD_PATH = os.path.join(
    _HELM_TEMPLATES, "grafana-dashboard-configmap.yaml"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dashboard_json():
    """Extract and parse the JSON from the Grafana dashboard configmap template."""
    with open(_GRAFANA_DASHBOARD_PATH, encoding="utf-8") as f:
        content = f.read()
    json_marker = "ocr-pipeline.json: |"
    idx = content.index(json_marker) + len(json_marker)
    json_text = content[idx:].strip()
    # Remove trailing Helm directives
    end_marker = "{{- end }}"
    end_idx = json_text.rfind(end_marker)
    if end_idx >= 0:
        json_text = json_text[:end_idx].strip()
    # Replace Go template escapes: {{ "{{foo}}" }} -> {{foo}}
    json_text = re.sub(r'\{\{ "(\{\{[^"]*\}\})" \}\}', r'\1', json_text)
    return json.loads(json_text)


@pytest.fixture(scope="module")
def all_panels(dashboard_json):
    """Return the flat list of all panels."""
    return dashboard_json["panels"]


@pytest.fixture(scope="module")
def template_variables(dashboard_json):
    """Return the list of template variables."""
    return dashboard_json.get("templating", {}).get("list", [])


# ---------------------------------------------------------------------------
# M-02: Cost per Tenant panel
# ---------------------------------------------------------------------------
class TestCostPerTenantPanel:
    """Validate the Cost per Tenant stat panel (ID 45)."""

    def test_panel_exists(self, all_panels):
        """Panel ID 45 exists in the dashboard."""
        ids = [p["id"] for p in all_panels]
        assert 45 in ids, "Expected panel ID 45 (Cost per Tenant)"

    def test_panel_type_is_stat(self, all_panels):
        """Cost per Tenant panel type is stat."""
        panel = next(p for p in all_panels if p["id"] == 45)
        assert panel["type"] == "stat"

    def test_panel_title_contains_cost(self, all_panels):
        """Panel title references cost."""
        panel = next(p for p in all_panels if p["id"] == 45)
        assert "cost" in panel["title"].lower()

    def test_panel_title_contains_tenant(self, all_panels):
        """Panel title references tenant."""
        panel = next(p for p in all_panels if p["id"] == 45)
        assert "tenant" in panel["title"].lower()

    def test_panel_has_description(self, all_panels):
        """Panel has a non-empty description."""
        panel = next(p for p in all_panels if p["id"] == 45)
        assert len(panel.get("description", "")) > 0

    def test_panel_queries_cost_metric(self, all_panels):
        """Panel queries ocr_cost_estimate_total metric."""
        panel = next(p for p in all_panels if p["id"] == 45)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("ocr_cost_estimate_total" in e for e in exprs)

    def test_panel_uses_tenant_id_filter(self, all_panels):
        """Panel queries use $tenant_id variable for filtering."""
        panel = next(p for p in all_panels if p["id"] == 45)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("$tenant_id" in e for e in exprs)

    def test_panel_unit_is_currency(self, all_panels):
        """Panel uses a currency unit."""
        panel = next(p for p in all_panels if p["id"] == 45)
        unit = panel["fieldConfig"]["defaults"].get("unit", "")
        assert "currency" in unit.lower()

    def test_panel_has_datasource(self, all_panels):
        """Panel datasource is prometheus."""
        panel = next(p for p in all_panels if p["id"] == 45)
        assert panel["datasource"]["type"] == "prometheus"


# ---------------------------------------------------------------------------
# M-03: Per-Tenant Processing Latency panel
# ---------------------------------------------------------------------------
class TestPerTenantLatencyPanel:
    """Validate the Per-Tenant Processing Latency timeseries panel (ID 46)."""

    def test_panel_exists(self, all_panels):
        """Panel ID 46 exists in the dashboard."""
        ids = [p["id"] for p in all_panels]
        assert 46 in ids, "Expected panel ID 46 (Per-Tenant Latency)"

    def test_panel_type_is_timeseries(self, all_panels):
        """Per-Tenant Latency panel type is timeseries."""
        panel = next(p for p in all_panels if p["id"] == 46)
        assert panel["type"] == "timeseries"

    def test_panel_title_contains_latency(self, all_panels):
        """Panel title references latency."""
        panel = next(p for p in all_panels if p["id"] == 46)
        assert "latency" in panel["title"].lower()

    def test_panel_has_description(self, all_panels):
        """Panel has a non-empty description."""
        panel = next(p for p in all_panels if p["id"] == 46)
        assert len(panel.get("description", "")) > 0

    def test_panel_has_p50_query(self, all_panels):
        """Panel has a histogram_quantile(0.5, ...) query for p50."""
        panel = next(p for p in all_panels if p["id"] == 46)
        exprs = [t["expr"] for t in panel["targets"]]
        p50_exprs = [e for e in exprs if "histogram_quantile(0.5" in e]
        assert len(p50_exprs) >= 1, "Expected p50 histogram_quantile query"

    def test_panel_has_p95_query(self, all_panels):
        """Panel has a histogram_quantile(0.95, ...) query for p95."""
        panel = next(p for p in all_panels if p["id"] == 46)
        exprs = [t["expr"] for t in panel["targets"]]
        p95_exprs = [e for e in exprs if "histogram_quantile(0.95" in e]
        assert len(p95_exprs) >= 1, "Expected p95 histogram_quantile query"

    def test_panel_queries_reference_duration_histogram(self, all_panels):
        """Queries reference ocr_processing_duration_seconds_bucket."""
        panel = next(p for p in all_panels if p["id"] == 46)
        for target in panel["targets"]:
            assert "ocr_processing_duration_seconds_bucket" in target["expr"]

    def test_panel_queries_use_tenant_id_filter(self, all_panels):
        """Queries use tenant_id label filter with $tenant_id variable."""
        panel = next(p for p in all_panels if p["id"] == 46)
        for target in panel["targets"]:
            assert "tenant_id" in target["expr"]
            assert "$tenant_id" in target["expr"]

    def test_panel_unit_is_seconds(self, all_panels):
        """Panel default unit is seconds."""
        panel = next(p for p in all_panels if p["id"] == 46)
        assert panel["fieldConfig"]["defaults"]["unit"] == "s"

    def test_panel_legend_contains_tenant(self, all_panels):
        """Legend format includes tenant_id."""
        panel = next(p for p in all_panels if p["id"] == 46)
        for target in panel["targets"]:
            assert "tenant_id" in target.get("legendFormat", "")

    def test_panel_has_datasource(self, all_panels):
        """Panel datasource is prometheus."""
        panel = next(p for p in all_panels if p["id"] == 46)
        assert panel["datasource"]["type"] == "prometheus"


# ---------------------------------------------------------------------------
# M-05: Queue Depth per Stage panel
# ---------------------------------------------------------------------------
class TestQueueDepthPerStagePanel:
    """Validate the Queue Depth per Stage timeseries panel (ID 47)."""

    def test_panel_exists(self, all_panels):
        """Panel ID 47 exists in the dashboard."""
        ids = [p["id"] for p in all_panels]
        assert 47 in ids, "Expected panel ID 47 (Queue Depth per Stage)"

    def test_panel_type_is_timeseries(self, all_panels):
        """Queue Depth panel type is timeseries."""
        panel = next(p for p in all_panels if p["id"] == 47)
        assert panel["type"] == "timeseries"

    def test_panel_title_contains_queue(self, all_panels):
        """Panel title references queue."""
        panel = next(p for p in all_panels if p["id"] == 47)
        assert "queue" in panel["title"].lower()

    def test_panel_has_description(self, all_panels):
        """Panel has a non-empty description."""
        panel = next(p for p in all_panels if p["id"] == 47)
        assert len(panel.get("description", "")) > 0

    def test_panel_has_extraction_stage(self, all_panels):
        """Panel has a target for the extraction queue stage."""
        panel = next(p for p in all_panels if p["id"] == 47)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("extraction" in e for e in exprs), "Missing extraction stage target"

    def test_panel_has_ocr_stage(self, all_panels):
        """Panel has a target for the OCR queue stage."""
        panel = next(p for p in all_panels if p["id"] == 47)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any('"ocr"' in e for e in exprs), "Missing OCR stage target"

    def test_panel_has_compression_stage(self, all_panels):
        """Panel has a target for the compression queue stage."""
        panel = next(p for p in all_panels if p["id"] == 47)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("compression" in e for e in exprs), "Missing compression stage target"

    def test_panel_has_nlp_stage(self, all_panels):
        """Panel has a target for the NLP queue stage."""
        panel = next(p for p in all_panels if p["id"] == 47)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("nlp" in e for e in exprs), "Missing NLP stage target"

    def test_panel_has_four_targets(self, all_panels):
        """Panel has exactly 4 targets (one per stage)."""
        panel = next(p for p in all_panels if p["id"] == 47)
        assert len(panel["targets"]) == 4

    def test_panel_uses_queue_depth_metric(self, all_panels):
        """All targets reference the ocr_queue_depth metric."""
        panel = next(p for p in all_panels if p["id"] == 47)
        for target in panel["targets"]:
            assert "ocr_queue_depth" in target["expr"]

    def test_panel_has_datasource(self, all_panels):
        """Panel datasource is prometheus."""
        panel = next(p for p in all_panels if p["id"] == 47)
        assert panel["datasource"]["type"] == "prometheus"

    def test_panel_legend_labels(self, all_panels):
        """Each target has a non-empty legendFormat."""
        panel = next(p for p in all_panels if p["id"] == 47)
        for target in panel["targets"]:
            assert len(target.get("legendFormat", "")) > 0


# ---------------------------------------------------------------------------
# M-08: Template variables
# ---------------------------------------------------------------------------
class TestTemplateVariables:
    """Validate template variables for tenant/time-range filtering."""

    def test_tenant_id_variable_exists(self, template_variables):
        """A 'tenant_id' template variable exists."""
        names = [v["name"] for v in template_variables]
        assert "tenant_id" in names, "Expected 'tenant_id' template variable"

    def test_tenant_id_variable_type_is_query(self, template_variables):
        """The tenant_id variable uses query type."""
        tv = next(v for v in template_variables if v["name"] == "tenant_id")
        assert tv["type"] == "query"

    def test_tenant_id_variable_includes_all(self, template_variables):
        """The tenant_id variable has includeAll enabled."""
        tv = next(v for v in template_variables if v["name"] == "tenant_id")
        assert tv["includeAll"] is True

    def test_tenant_id_variable_is_multi(self, template_variables):
        """The tenant_id variable allows multi-select."""
        tv = next(v for v in template_variables if v["name"] == "tenant_id")
        assert tv["multi"] is True

    def test_tenant_id_variable_queries_label_values(self, template_variables):
        """The tenant_id query uses label_values() to discover tenant IDs."""
        tv = next(v for v in template_variables if v["name"] == "tenant_id")
        assert "label_values(" in tv.get("query", "")
        assert "tenant_id" in tv.get("query", "")

    def test_tenant_id_variable_has_datasource(self, template_variables):
        """The tenant_id variable has a prometheus datasource."""
        tv = next(v for v in template_variables if v["name"] == "tenant_id")
        assert tv.get("datasource", {}).get("type") == "prometheus"

    def test_interval_variable_has_6h(self, template_variables):
        """The interval variable includes 6h option."""
        tv = next(v for v in template_variables if v["name"] == "interval")
        option_values = [o["value"] for o in tv.get("options", [])]
        assert "6h" in option_values, "Expected 6h in interval options"

    def test_interval_variable_has_24h(self, template_variables):
        """The interval variable includes 24h option."""
        tv = next(v for v in template_variables if v["name"] == "interval")
        option_values = [o["value"] for o in tv.get("options", [])]
        assert "24h" in option_values, "Expected 24h in interval options"

    def test_interval_query_includes_6h_24h(self, template_variables):
        """The interval query string includes 6h and 24h."""
        tv = next(v for v in template_variables if v["name"] == "interval")
        query = tv.get("query", "")
        assert "6h" in query
        assert "24h" in query

    def test_three_template_variables(self, template_variables):
        """Dashboard has exactly 3 template variables (interval, tenant, tenant_id)."""
        assert len(template_variables) == 3, (
            f"Expected 3 template variables, got {len(template_variables)}: "
            f"{[v['name'] for v in template_variables]}"
        )

    def test_existing_tenant_variable_preserved(self, template_variables):
        """The original 'tenant' variable still exists."""
        names = [v["name"] for v in template_variables]
        assert "tenant" in names

    def test_existing_interval_variable_preserved(self, template_variables):
        """The original 'interval' variable still exists."""
        names = [v["name"] for v in template_variables]
        assert "interval" in names


# ---------------------------------------------------------------------------
# Overall dashboard integrity
# ---------------------------------------------------------------------------
class TestDashboardIntegrity:
    """Validate overall dashboard structure after the batch additions."""

    def test_total_panel_count(self, all_panels):
        """Dashboard has 53 panels (50 existing + 3 new: GPU VRAM + CPU/GPU engine + tenant storage)."""
        assert len(all_panels) == 53, (
            f"Expected 53 panels, got {len(all_panels)}"
        )

    def test_all_panel_ids_unique(self, all_panels):
        """All panel IDs are unique."""
        ids = [p["id"] for p in all_panels]
        assert len(ids) == len(set(ids)), f"Duplicate panel IDs: {ids}"

    def test_new_row_panel_exists(self, all_panels):
        """The new 'Cost & Queue Analysis' row panel exists (ID 44)."""
        row = next((p for p in all_panels if p["id"] == 44), None)
        assert row is not None, "Expected row panel ID 44"
        assert row["type"] == "row"

    def test_new_panel_ids_are_sequential(self, all_panels):
        """New panel IDs 44-47 are present."""
        ids = {p["id"] for p in all_panels}
        for expected_id in (44, 45, 46, 47):
            assert expected_id in ids, f"Missing panel ID {expected_id}"

    def test_dashboard_json_is_valid(self, dashboard_json):
        """Dashboard JSON parses and has required top-level keys."""
        assert isinstance(dashboard_json, dict)
        assert "panels" in dashboard_json
        assert "templating" in dashboard_json
        assert "title" in dashboard_json
