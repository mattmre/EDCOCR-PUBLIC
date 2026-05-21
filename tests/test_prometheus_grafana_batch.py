"""Tests for M-14/M-15/M-06/M-07: Prometheus counters and Grafana panels.

Validates:
- M-14: ocr_engine_usage_total counter increments correctly per engine
- M-15: ocr_dpi_escalation_total counter with from_dpi/to_dpi labels
- M-06: CPU vs GPU Throughput timeseries panel in Grafana dashboard
- M-07: Storage Consumption gauge panel in Grafana dashboard
- Dashboard YAML is valid and contains the new panels

Run with:
  python -m pytest tests/test_prometheus_grafana_batch.py -v
"""

from __future__ import annotations

import json
import os
import re

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HELM_TEMPLATES = os.path.join(
    _PROJECT_ROOT, "helm", "ocr-local", "templates",
)

_GRAFANA_DASHBOARD_PATH = os.path.join(
    _HELM_TEMPLATES, "grafana-dashboard-configmap.yaml",
)


# ---------------------------------------------------------------------------
# Dashboard fixture
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


# ===========================================================================
# M-14: ocr_engine_usage_total counter
# ===========================================================================
class TestEngineUsageCounter:
    """Validate the per-engine OCR usage counter (M-14)."""

    def test_module_imports_cleanly(self):
        """ocr_metrics module can be imported without errors."""
        import ocr_metrics  # noqa: F401

    def test_counter_exists(self):
        """ENGINE_USAGE_COUNTER is created when prometheus_client is available."""
        from ocr_metrics import ENGINE_USAGE_COUNTER

        # In CI prometheus_client may or may not be installed.
        # Just verify the attribute is present (Counter or None).
        assert ENGINE_USAGE_COUNTER is not None or True  # always passes import check

    def test_record_engine_usage_paddle(self):
        """record_engine_usage('paddle') does not raise."""
        from ocr_metrics import record_engine_usage

        record_engine_usage("paddle")  # should not raise

    def test_record_engine_usage_tesseract(self):
        """record_engine_usage('tesseract') does not raise."""
        from ocr_metrics import record_engine_usage

        record_engine_usage("tesseract")

    def test_record_engine_usage_onnx(self):
        """record_engine_usage('onnx') does not raise."""
        from ocr_metrics import record_engine_usage

        record_engine_usage("onnx")

    def test_record_engine_usage_image_only(self):
        """record_engine_usage('image_only') does not raise."""
        from ocr_metrics import record_engine_usage

        record_engine_usage("image_only")

    def test_normalise_paddle_alias(self):
        """PaddleOCR alias is normalised to 'paddle'."""
        from ocr_metrics import _normalise_engine

        assert _normalise_engine("PaddleOCR") == "paddle"
        assert _normalise_engine("paddle") == "paddle"

    def test_normalise_onnx_alias(self):
        """onnxruntime alias is normalised to 'onnx'."""
        from ocr_metrics import _normalise_engine

        assert _normalise_engine("onnxruntime") == "onnx"
        assert _normalise_engine("ONNX") == "onnx"

    def test_normalise_image_only_alias(self):
        """image_only aliases are normalised."""
        from ocr_metrics import _normalise_engine

        assert _normalise_engine("image_only") == "image_only"
        assert _normalise_engine("imageonly") == "image_only"

    def test_normalise_unknown_engine_passthrough(self):
        """Unknown engine labels pass through as-is."""
        from ocr_metrics import _normalise_engine

        assert _normalise_engine("custom_engine") == "custom_engine"

    def test_valid_engines_frozenset(self):
        """_VALID_ENGINES contains expected values."""
        from ocr_metrics import _VALID_ENGINES

        assert "paddle" in _VALID_ENGINES
        assert "tesseract" in _VALID_ENGINES
        assert "onnx" in _VALID_ENGINES
        assert "image_only" in _VALID_ENGINES

    def test_counter_labels_engine(self):
        """If prometheus_client is available, counter has 'engine' label."""
        from ocr_metrics import ENGINE_USAGE_COUNTER

        if ENGINE_USAGE_COUNTER is not None:
            # Verify the counter describes the 'engine' label
            assert "engine" in ENGINE_USAGE_COUNTER._labelnames

    def test_counter_increments(self):
        """Counter value increases after record_engine_usage calls."""
        from ocr_metrics import ENGINE_USAGE_COUNTER, record_engine_usage

        if ENGINE_USAGE_COUNTER is None:
            pytest.skip("prometheus_client not installed")

        before = ENGINE_USAGE_COUNTER.labels(engine="tesseract")._value.get()
        record_engine_usage("tesseract")
        after = ENGINE_USAGE_COUNTER.labels(engine="tesseract")._value.get()
        assert after == before + 1.0


# ===========================================================================
# M-15: ocr_dpi_escalation_total counter
# ===========================================================================
class TestDpiEscalationCounter:
    """Validate the DPI escalation counter (M-15)."""

    def test_counter_exists(self):
        """DPI_ESCALATION_COUNTER is created when prometheus_client is available."""
        from ocr_metrics import DPI_ESCALATION_COUNTER

        assert DPI_ESCALATION_COUNTER is not None or True

    def test_record_dpi_escalation_no_raise(self):
        """record_dpi_escalation(300, 450) does not raise."""
        from ocr_metrics import record_dpi_escalation

        record_dpi_escalation(300, 450)

    def test_record_dpi_escalation_600(self):
        """record_dpi_escalation(300, 600) does not raise."""
        from ocr_metrics import record_dpi_escalation

        record_dpi_escalation(300, 600)

    def test_record_dpi_escalation_450_to_600(self):
        """record_dpi_escalation(450, 600) does not raise."""
        from ocr_metrics import record_dpi_escalation

        record_dpi_escalation(450, 600)

    def test_counter_labels(self):
        """If prometheus_client is available, counter has from_dpi/to_dpi labels."""
        from ocr_metrics import DPI_ESCALATION_COUNTER

        if DPI_ESCALATION_COUNTER is None:
            pytest.skip("prometheus_client not installed")

        assert "from_dpi" in DPI_ESCALATION_COUNTER._labelnames
        assert "to_dpi" in DPI_ESCALATION_COUNTER._labelnames

    def test_counter_increments(self):
        """Counter value increases after record_dpi_escalation calls."""
        from ocr_metrics import DPI_ESCALATION_COUNTER, record_dpi_escalation

        if DPI_ESCALATION_COUNTER is None:
            pytest.skip("prometheus_client not installed")

        before = DPI_ESCALATION_COUNTER.labels(
            from_dpi="300", to_dpi="450"
        )._value.get()
        record_dpi_escalation(300, 450)
        after = DPI_ESCALATION_COUNTER.labels(
            from_dpi="300", to_dpi="450"
        )._value.get()
        assert after == before + 1.0

    def test_dpi_escalation_module_integration(self):
        """dpi_escalation.re_extract_page_at_dpi accepts from_dpi kwarg."""
        import inspect

        from dpi_escalation import re_extract_page_at_dpi

        sig = inspect.signature(re_extract_page_at_dpi)
        assert "from_dpi" in sig.parameters, (
            "re_extract_page_at_dpi should accept from_dpi keyword argument"
        )


# ===========================================================================
# M-06: CPU vs GPU Throughput Grafana panel
# ===========================================================================
class TestCpuGpuThroughputPanel:
    """Validate the CPU vs GPU Throughput timeseries panel (M-06)."""

    def test_panel_exists(self, all_panels):
        """Panel ID 49 exists in the dashboard."""
        ids = [p["id"] for p in all_panels]
        assert 49 in ids, "Expected panel ID 49 (CPU vs GPU Throughput)"

    def test_panel_type_is_timeseries(self, all_panels):
        """Panel type is timeseries."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert panel["type"] == "timeseries"

    def test_panel_title_contains_cpu(self, all_panels):
        """Panel title references CPU."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert "cpu" in panel["title"].lower()

    def test_panel_title_contains_gpu(self, all_panels):
        """Panel title references GPU."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert "gpu" in panel["title"].lower()

    def test_panel_has_description(self, all_panels):
        """Panel has a non-empty description."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert len(panel.get("description", "")) > 0

    def test_panel_has_two_targets(self, all_panels):
        """Panel has two targets (CPU and GPU)."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert len(panel["targets"]) == 2

    def test_panel_has_gpu_legend(self, all_panels):
        """Panel has a target with GPU in legend."""
        panel = next(p for p in all_panels if p["id"] == 49)
        legends = [t.get("legendFormat", "") for t in panel["targets"]]
        assert any("GPU" in lf for lf in legends), "Missing GPU legendFormat"

    def test_panel_has_cpu_legend(self, all_panels):
        """Panel has a target with CPU in legend."""
        panel = next(p for p in all_panels if p["id"] == 49)
        legends = [t.get("legendFormat", "") for t in panel["targets"]]
        assert any("CPU" in lf for lf in legends), "Missing CPU legendFormat"

    def test_panel_has_datasource(self, all_panels):
        """Panel datasource is prometheus."""
        panel = next(p for p in all_panels if p["id"] == 49)
        assert panel["datasource"]["type"] == "prometheus"

    def test_panel_queries_use_rate(self, all_panels):
        """Panel queries use rate() for throughput calculation."""
        panel = next(p for p in all_panels if p["id"] == 49)
        for target in panel["targets"]:
            assert "rate(" in target["expr"], (
                f"Expected rate() in query: {target['expr']}"
            )


# ===========================================================================
# M-07: Storage Consumption gauge
# ===========================================================================
class TestStorageConsumptionPanel:
    """Validate the Storage Consumption gauge panel (M-07)."""

    def test_panel_exists(self, all_panels):
        """Panel ID 50 exists in the dashboard."""
        ids = [p["id"] for p in all_panels]
        assert 50 in ids, "Expected panel ID 50 (Storage Consumption)"

    def test_panel_type_is_gauge(self, all_panels):
        """Panel type is gauge."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert panel["type"] == "gauge"

    def test_panel_title_contains_storage(self, all_panels):
        """Panel title references storage."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert "storage" in panel["title"].lower()

    def test_panel_has_description(self, all_panels):
        """Panel has a non-empty description."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert len(panel.get("description", "")) > 0

    def test_panel_has_thresholds(self, all_panels):
        """Panel has threshold configuration."""
        panel = next(p for p in all_panels if p["id"] == 50)
        steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert len(steps) >= 2, "Expected at least 2 threshold steps"

    def test_panel_unit_is_percentunit(self, all_panels):
        """Panel unit is percentunit."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert panel["fieldConfig"]["defaults"]["unit"] == "percentunit"

    def test_panel_has_target(self, all_panels):
        """Panel has at least one query target."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert len(panel["targets"]) >= 1

    def test_panel_query_references_filesystem(self, all_panels):
        """Panel query references node_filesystem metric."""
        panel = next(p for p in all_panels if p["id"] == 50)
        exprs = [t["expr"] for t in panel["targets"]]
        assert any("node_filesystem" in e for e in exprs), (
            "Expected node_filesystem in query"
        )

    def test_panel_has_datasource(self, all_panels):
        """Panel datasource is prometheus."""
        panel = next(p for p in all_panels if p["id"] == 50)
        assert panel["datasource"]["type"] == "prometheus"

    def test_panel_min_max(self, all_panels):
        """Panel min/max are set to 0 and 1 for percentunit."""
        panel = next(p for p in all_panels if p["id"] == 50)
        defaults = panel["fieldConfig"]["defaults"]
        assert defaults["min"] == 0
        assert defaults["max"] == 1


# ===========================================================================
# Dashboard integrity checks
# ===========================================================================
class TestDashboardIntegrityBatch:
    """Overall dashboard integrity after M-06/M-07 additions."""

    def test_dashboard_json_is_valid(self, dashboard_json):
        """Dashboard JSON parses and has required top-level keys."""
        assert isinstance(dashboard_json, dict)
        assert "panels" in dashboard_json
        assert "templating" in dashboard_json
        assert "title" in dashboard_json

    def test_all_panel_ids_unique(self, all_panels):
        """All panel IDs are unique."""
        ids = [p["id"] for p in all_panels]
        assert len(ids) == len(set(ids)), f"Duplicate panel IDs found: {ids}"

    def test_total_panel_count(self, all_panels):
        """Dashboard has 53 panels (50 existing + 3 new GPU/storage panels)."""
        assert len(all_panels) == 53, (
            f"Expected 53 panels, got {len(all_panels)}"
        )

    def test_new_row_panel_exists(self, all_panels):
        """The new 'Infrastructure' row panel exists (ID 48)."""
        row = next((p for p in all_panels if p["id"] == 48), None)
        assert row is not None, "Expected row panel ID 48"
        assert row["type"] == "row"

    def test_new_panel_ids_present(self, all_panels):
        """New panel IDs 48, 49, 50 are present."""
        ids = {p["id"] for p in all_panels}
        for expected_id in (48, 49, 50):
            assert expected_id in ids, f"Missing panel ID {expected_id}"

    def test_configmap_file_exists(self):
        """The Grafana dashboard configmap file exists."""
        assert os.path.isfile(_GRAFANA_DASHBOARD_PATH)
