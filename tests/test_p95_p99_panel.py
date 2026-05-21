"""Tests for M-04: p95/p99 processing time Grafana panel.

Validates:
- Grafana dashboard JSON has the new p95/p99 panel (ID 43)
- Panel type is timeseries with unit seconds
- Panel uses histogram_quantile(0.95, ...) and histogram_quantile(0.99, ...)
- Panel targets reference ocr_processing_duration_seconds_bucket
- Panel has proper title, description, and thresholds

Run with:
  python -m pytest tests/test_p95_p99_panel.py -v
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
    """Extract and parse the JSON from the Grafana dashboard configmap template.

    Follows the same extraction pattern as test_sla_breach_panel.py.
    """
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
def p95_p99_panel(dashboard_json):
    """Return the p95/p99 panel (ID 43) from the dashboard."""
    panels = dashboard_json["panels"]
    matches = [p for p in panels if p.get("id") == 43]
    assert len(matches) == 1, "Expected exactly one panel with ID 43"
    return matches[0]


# ---------------------------------------------------------------------------
# Dashboard structure tests
# ---------------------------------------------------------------------------
class TestDashboardHasP95P99Panel:
    """Validate the p95/p99 panel exists in the Grafana dashboard."""

    def test_panel_exists(self, dashboard_json):
        """A panel with ID 43 exists in the dashboard."""
        ids = [p["id"] for p in dashboard_json["panels"]]
        assert 43 in ids

    def test_panel_ids_unique(self, dashboard_json):
        """All panel IDs are unique across the dashboard."""
        panels = dashboard_json["panels"]
        ids = [p["id"] for p in panels]
        assert len(ids) == len(set(ids)), f"Duplicate panel IDs found: {ids}"

    def test_total_panel_count(self, dashboard_json):
        """Dashboard has 53 panels (50 existing + 3 new: GPU VRAM + CPU/GPU engine + tenant storage)."""
        panels = dashboard_json["panels"]
        assert len(panels) == 53, (
            f"Expected 53 panels, got {len(panels)}"
        )


# ---------------------------------------------------------------------------
# Panel type and metadata tests
# ---------------------------------------------------------------------------
class TestP95P99PanelMetadata:
    """Validate panel type, title, and configuration."""

    def test_panel_type_is_timeseries(self, p95_p99_panel):
        """Panel type is timeseries."""
        assert p95_p99_panel["type"] == "timeseries"

    def test_panel_title(self, p95_p99_panel):
        """Panel title contains p95 and p99."""
        title = p95_p99_panel["title"]
        assert "p95" in title.lower()
        assert "p99" in title.lower()

    def test_panel_has_description(self, p95_p99_panel):
        """Panel has a non-empty description."""
        desc = p95_p99_panel.get("description", "")
        assert len(desc) > 0, "Panel should have a description"

    def test_panel_has_datasource(self, p95_p99_panel):
        """Panel datasource is prometheus."""
        ds = p95_p99_panel.get("datasource", {})
        assert ds.get("type") == "prometheus"


# ---------------------------------------------------------------------------
# Unit and field config tests
# ---------------------------------------------------------------------------
class TestP95P99PanelUnits:
    """Validate the panel uses seconds as the unit."""

    def test_default_unit_is_seconds(self, p95_p99_panel):
        """Field config defaults unit is 's' (seconds)."""
        defaults = p95_p99_panel["fieldConfig"]["defaults"]
        assert defaults["unit"] == "s"

    def test_has_thresholds(self, p95_p99_panel):
        """Panel has threshold steps defined."""
        defaults = p95_p99_panel["fieldConfig"]["defaults"]
        thresholds = defaults.get("thresholds", {})
        steps = thresholds.get("steps", [])
        assert len(steps) >= 2, "Expected at least 2 threshold steps"

    def test_draw_style_is_line(self, p95_p99_panel):
        """Panel uses line draw style."""
        custom = p95_p99_panel["fieldConfig"]["defaults"].get("custom", {})
        assert custom.get("drawStyle") == "line"


# ---------------------------------------------------------------------------
# PromQL query tests
# ---------------------------------------------------------------------------
class TestP95P99PanelQueries:
    """Validate histogram_quantile PromQL queries."""

    def test_has_two_targets(self, p95_p99_panel):
        """Panel has exactly two query targets (p95 and p99)."""
        targets = p95_p99_panel["targets"]
        assert len(targets) == 2

    def test_has_p95_histogram_quantile_query(self, p95_p99_panel):
        """One target uses histogram_quantile(0.95, ...)."""
        targets = p95_p99_panel["targets"]
        p95_targets = [
            t for t in targets
            if "histogram_quantile(0.95" in t["expr"]
        ]
        assert len(p95_targets) == 1, "Expected exactly one p95 query"

    def test_has_p99_histogram_quantile_query(self, p95_p99_panel):
        """One target uses histogram_quantile(0.99, ...)."""
        targets = p95_p99_panel["targets"]
        p99_targets = [
            t for t in targets
            if "histogram_quantile(0.99" in t["expr"]
        ]
        assert len(p99_targets) == 1, "Expected exactly one p99 query"

    def test_p95_query_references_histogram_bucket(self, p95_p99_panel):
        """p95 query references ocr_processing_duration_seconds_bucket."""
        targets = p95_p99_panel["targets"]
        p95_target = next(
            t for t in targets if "histogram_quantile(0.95" in t["expr"]
        )
        assert "ocr_processing_duration_seconds_bucket" in p95_target["expr"]

    def test_p99_query_references_histogram_bucket(self, p95_p99_panel):
        """p99 query references ocr_processing_duration_seconds_bucket."""
        targets = p95_p99_panel["targets"]
        p99_target = next(
            t for t in targets if "histogram_quantile(0.99" in t["expr"]
        )
        assert "ocr_processing_duration_seconds_bucket" in p99_target["expr"]

    def test_p95_legend_format(self, p95_p99_panel):
        """p95 target has legendFormat containing 'p95'."""
        targets = p95_p99_panel["targets"]
        p95_target = next(
            t for t in targets if "histogram_quantile(0.95" in t["expr"]
        )
        assert "p95" in p95_target["legendFormat"].lower()

    def test_p99_legend_format(self, p95_p99_panel):
        """p99 target has legendFormat containing 'p99'."""
        targets = p95_p99_panel["targets"]
        p99_target = next(
            t for t in targets if "histogram_quantile(0.99" in t["expr"]
        )
        assert "p99" in p99_target["legendFormat"].lower()

    def test_queries_use_rate_interval(self, p95_p99_panel):
        """Both queries use $__rate_interval for proper rate calculation."""
        for target in p95_p99_panel["targets"]:
            assert "$__rate_interval" in target["expr"], (
                f"Query should use $__rate_interval: {target['expr']}"
            )

    def test_queries_use_sum_by_le(self, p95_p99_panel):
        """Both queries aggregate with sum(...) by (le) for histogram_quantile."""
        for target in p95_p99_panel["targets"]:
            assert "sum(rate(" in target["expr"]
            assert "by (le)" in target["expr"]
