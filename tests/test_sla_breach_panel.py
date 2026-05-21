"""Tests for M-01: SLA breach rate Grafana panel and alert rule.

Validates:
- OCRSLABreachRate alert exists in PrometheusRule YAML
- ocr:sla_compliance_rate recording rule exists
- Grafana dashboard has SLA Compliance row and panels (IDs 40-42)
- SLA compliance gauge has correct thresholds (red < 95, yellow < 99, green >= 99)
- SLA breach history panel references both latency and error rate metrics

Run with:
  python -m pytest tests/test_sla_breach_panel.py -v
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

_PROMETHEUSRULE_PATH = os.path.join(_HELM_TEMPLATES, "prometheusrule.yaml")
_GRAFANA_DASHBOARD_PATH = os.path.join(
    _HELM_TEMPLATES, "grafana-dashboard-configmap.yaml"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prometheusrule_content():
    """Read the raw PrometheusRule YAML template content."""
    with open(_PROMETHEUSRULE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def dashboard_json():
    """Extract and parse the JSON from the Grafana dashboard configmap template.

    Follows the same extraction pattern as test_tenant_metrics.py.
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


# ---------------------------------------------------------------------------
# PrometheusRule alert tests
# ---------------------------------------------------------------------------
class TestOCRSLABreachAlert:
    """Validate OCRSLABreachRate alert rule in PrometheusRule YAML."""

    def test_alert_exists(self, prometheusrule_content):
        """OCRSLABreachRate alert is defined."""
        assert "OCRSLABreachRate" in prometheusrule_content

    def test_alert_severity_warning(self, prometheusrule_content):
        """Alert has severity: warning label."""
        # Find the OCRSLABreachRate block and check severity
        idx = prometheusrule_content.index("OCRSLABreachRate")
        block = prometheusrule_content[idx:idx + 500]
        assert "severity: warning" in block

    def test_alert_for_duration(self, prometheusrule_content):
        """Alert fires after 5 minutes."""
        idx = prometheusrule_content.index("OCRSLABreachRate")
        block = prometheusrule_content[idx:idx + 500]
        assert "for: 5m" in block

    def test_alert_checks_p95_latency(self, prometheusrule_content):
        """Alert expression checks P95 latency via histogram_quantile."""
        idx = prometheusrule_content.index("OCRSLABreachRate")
        block = prometheusrule_content[idx:idx + 600]
        assert "histogram_quantile(0.95" in block
        assert "ocr_processing_duration_seconds_bucket" in block

    def test_alert_checks_error_rate(self, prometheusrule_content):
        """Alert expression checks error rate SLO."""
        idx = prometheusrule_content.index("OCRSLABreachRate")
        block = prometheusrule_content[idx:idx + 600]
        assert "ocr_job_error_rate_1h" in block
        assert "0.01" in block

    def test_alert_has_annotations(self, prometheusrule_content):
        """Alert has summary and description annotations."""
        idx = prometheusrule_content.index("OCRSLABreachRate")
        block = prometheusrule_content[idx:idx + 600]
        assert "summary:" in block
        assert "description:" in block
        assert "SLA breach" in block


# ---------------------------------------------------------------------------
# PrometheusRule recording rule tests
# ---------------------------------------------------------------------------
class TestSLAComplianceRecordingRule:
    """Validate ocr:sla_compliance_rate recording rule."""

    def test_recording_rule_exists(self, prometheusrule_content):
        """ocr:sla_compliance_rate recording rule is defined."""
        assert "ocr:sla_compliance_rate" in prometheusrule_content

    def test_recording_rule_uses_histogram(self, prometheusrule_content):
        """Recording rule references the processing duration histogram."""
        idx = prometheusrule_content.index("ocr:sla_compliance_rate")
        block = prometheusrule_content[idx:idx + 400]
        assert "histogram_quantile" in block
        assert "ocr_processing_duration_seconds_bucket" in block

    def test_recording_rule_uses_error_rate(self, prometheusrule_content):
        """Recording rule references the error rate metric."""
        idx = prometheusrule_content.index("ocr:sla_compliance_rate")
        block = prometheusrule_content[idx:idx + 400]
        assert "ocr_job_error_rate_1h" in block

    def test_recording_rule_uses_bool_comparison(self, prometheusrule_content):
        """Recording rule uses bool modifier for binary comparison."""
        idx = prometheusrule_content.index("ocr:sla_compliance_rate")
        block = prometheusrule_content[idx:idx + 400]
        assert "> bool 30" in block
        assert "> bool 0.01" in block


# ---------------------------------------------------------------------------
# Grafana dashboard panel tests
# ---------------------------------------------------------------------------
class TestGrafanaSLAPanels:
    """Validate SLA Compliance panels in Grafana dashboard JSON."""

    def test_dashboard_is_valid_json(self, dashboard_json):
        """Dashboard JSON parses without errors."""
        assert isinstance(dashboard_json, dict)
        assert "panels" in dashboard_json

    def test_sla_compliance_row_exists(self, dashboard_json):
        """An 'SLA Compliance' row panel exists."""
        panels = dashboard_json["panels"]
        row_panels = [
            p for p in panels
            if p.get("type") == "row" and "SLA Compliance" in p.get("title", "")
        ]
        assert len(row_panels) == 1, "Expected exactly one 'SLA Compliance' row panel"

    def test_sla_compliance_row_id(self, dashboard_json):
        """SLA Compliance row has panel ID 40."""
        panels = dashboard_json["panels"]
        row = next(
            p for p in panels
            if p.get("type") == "row" and "SLA Compliance" in p.get("title", "")
        )
        assert row["id"] == 40

    def test_sla_compliance_rate_gauge_exists(self, dashboard_json):
        """SLA Compliance Rate gauge panel exists with ID 41."""
        panels = dashboard_json["panels"]
        gauge_panels = [
            p for p in panels
            if p.get("id") == 41 and p.get("type") == "gauge"
        ]
        assert len(gauge_panels) == 1
        assert "SLA Compliance Rate" in gauge_panels[0]["title"]

    def test_sla_compliance_gauge_thresholds(self, dashboard_json):
        """SLA gauge has correct threshold steps: red < 95, yellow < 99, green >= 99."""
        panels = dashboard_json["panels"]
        gauge = next(p for p in panels if p.get("id") == 41)
        steps = gauge["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert len(steps) == 3
        # Step 0: red (base, value null)
        assert steps[0]["color"] == "red"
        assert steps[0]["value"] is None
        # Step 1: yellow at 95
        assert steps[1]["color"] == "yellow"
        assert steps[1]["value"] == 95
        # Step 2: green at 99
        assert steps[2]["color"] == "green"
        assert steps[2]["value"] == 99

    def test_sla_compliance_gauge_unit(self, dashboard_json):
        """SLA gauge uses percent unit with 0-100 range."""
        panels = dashboard_json["panels"]
        gauge = next(p for p in panels if p.get("id") == 41)
        defaults = gauge["fieldConfig"]["defaults"]
        assert defaults["unit"] == "percent"
        assert defaults["min"] == 0
        assert defaults["max"] == 100

    def test_sla_compliance_gauge_query(self, dashboard_json):
        """SLA gauge queries ocr:sla_compliance_rate * 100."""
        panels = dashboard_json["panels"]
        gauge = next(p for p in panels if p.get("id") == 41)
        targets = gauge["targets"]
        assert len(targets) >= 1
        assert "ocr:sla_compliance_rate * 100" in targets[0]["expr"]

    def test_sla_breach_history_panel_exists(self, dashboard_json):
        """SLA Breach History timeseries panel exists with ID 42."""
        panels = dashboard_json["panels"]
        ts_panels = [
            p for p in panels
            if p.get("id") == 42 and p.get("type") == "timeseries"
        ]
        assert len(ts_panels) == 1
        assert "SLA Breach History" in ts_panels[0]["title"]

    def test_sla_breach_history_has_latency_query(self, dashboard_json):
        """Breach history panel queries P95 latency from histogram."""
        panels = dashboard_json["panels"]
        panel = next(p for p in panels if p.get("id") == 42)
        exprs = [t["expr"] for t in panel["targets"]]
        latency_exprs = [
            e for e in exprs
            if "histogram_quantile(0.95" in e
            and "ocr_processing_duration_seconds_bucket" in e
        ]
        assert len(latency_exprs) >= 1, "Expected P95 latency query"

    def test_sla_breach_history_has_error_rate_query(self, dashboard_json):
        """Breach history panel queries error rate."""
        panels = dashboard_json["panels"]
        panel = next(p for p in panels if p.get("id") == 42)
        exprs = [t["expr"] for t in panel["targets"]]
        error_exprs = [e for e in exprs if "ocr_job_error_rate_1h" in e]
        assert len(error_exprs) >= 1, "Expected error rate query"

    def test_sla_breach_history_has_threshold_overrides(self, dashboard_json):
        """Breach history panel has field overrides with SLO threshold lines."""
        panels = dashboard_json["panels"]
        panel = next(p for p in panels if p.get("id") == 42)
        overrides = panel["fieldConfig"].get("overrides", [])
        assert len(overrides) >= 2, "Expected at least 2 field overrides (latency + error)"

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

    def test_sla_panels_have_descriptions(self, dashboard_json):
        """SLA content panels (41, 42) have non-empty descriptions."""
        panels = dashboard_json["panels"]
        for panel_id in (41, 42):
            panel = next(p for p in panels if p.get("id") == panel_id)
            desc = panel.get("description", "")
            assert len(desc) > 0, f"Panel {panel_id} has no description"
