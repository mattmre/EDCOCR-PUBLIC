"""Tests for tenant-scoped Prometheus metrics (M-09+M-23).

Covers:
- tenant_id field on Job model
- 4 new tenant metric families in PipelineCollector
- Empty tenant_id exclusion
- Multi-tenant metric separation
- Grafana dashboard JSON structure validation

Run with:
  DJANGO_SETTINGS_MODULE=coordinator.settings_test python -m pytest tests/test_tenant_metrics.py -v
"""

import importlib.util
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Add coordinator to sys.path for model imports.  See .
# ---------------------------------------------------------------------------
_coordinator_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coordinator"
)
if _coordinator_dir not in sys.path:
    sys.path.insert(0, _coordinator_dir)

_HAS_DJANGO = False
if importlib.util.find_spec("django") is not None:
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        try:
            import django

            django.setup()
            _HAS_DJANGO = True
        except Exception:
            pass

_skip_no_django = pytest.mark.skipif(
    not _HAS_DJANGO, reason="Django not configured"
)


# ---------------------------------------------------------------------------
# Job model field tests
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestJobTenantIdField:
    """Verify tenant_id field exists and behaves correctly on Job model."""

    def test_tenant_id_default_empty_string(self):
        from jobs.models import Job

        job = Job.objects.create(source_file="/test.pdf")
        assert job.tenant_id == ""

    def test_tenant_id_set_on_create(self):
        from jobs.models import Job

        job = Job.objects.create(
            source_file="/test.pdf", tenant_id="acme-corp"
        )
        job.refresh_from_db()
        assert job.tenant_id == "acme-corp"

    def test_tenant_id_max_length(self):
        from jobs.models import Job

        long_id = "t" * 128
        job = Job.objects.create(
            source_file="/test.pdf", tenant_id=long_id
        )
        job.refresh_from_db()
        assert job.tenant_id == long_id

    def test_tenant_id_filterable(self):
        from jobs.models import Job

        Job.objects.create(source_file="/a.pdf", tenant_id="tenant-a")
        Job.objects.create(source_file="/b.pdf", tenant_id="tenant-b")
        Job.objects.create(source_file="/c.pdf", tenant_id="")

        assert Job.objects.filter(tenant_id="tenant-a").count() == 1
        assert Job.objects.exclude(tenant_id="").count() == 2


# ---------------------------------------------------------------------------
# Tenant-scoped metric family tests
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestTenantMetricFamiliesPresent:
    """Verify tenant metric families appear in collect() and describe()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_describe_yields_22_descriptors(self):
        """describe() yields 22 descriptors (17 original + 4 tenant + 1 histogram)."""
        descriptors = list(self.collector.describe())
        assert len(descriptors) == 22, (
            f"Expected 22, got {len(descriptors)}: "
            f"{[d.name for d in descriptors]}"
        )

    def test_collect_yields_22_metric_families(self):
        """collect() yields 22 metric families (17 original + 4 tenant + 1 histogram)."""
        families = list(self.collector.collect())
        family_names = [f.name for f in families]
        assert len(families) == 22, (
            f"Expected 22, got {len(families)}: {family_names}"
        )

    def test_tenant_metric_names_in_describe(self):
        """All 4 tenant metric names appear in describe() output."""
        descriptor_names = [d.name for d in self.collector.describe()]
        expected = [
            "ocr_tenant_jobs_total",
            "ocr_tenant_pages_processed",
            "ocr_tenant_error_rate",
            "ocr_tenant_processing_time_avg_ms",
        ]
        for name in expected:
            assert name in descriptor_names, (
                f"Expected descriptor '{name}' not found"
            )

    def test_tenant_metric_names_in_collect(self):
        """All 4 tenant metric names appear in collect() output."""
        families = {f.name: f for f in self.collector.collect()}
        expected = [
            "ocr_tenant_jobs_total",
            "ocr_tenant_pages_processed",
            "ocr_tenant_error_rate",
            "ocr_tenant_processing_time_avg_ms",
        ]
        for name in expected:
            assert name in families, (
                f"Expected metric '{name}' not found"
            )


@_skip_no_django
@pytest.mark.django_db
class TestTenantMetricsWithData:
    """Verify tenant metrics produce correct values with job data."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_tenant_jobs_by_status(self):
        """ocr_tenant_jobs_total shows job counts per tenant and status."""
        self.Job.objects.create(
            source_file="/a.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.COMPLETED,
        )
        self.Job.objects.create(
            source_file="/b.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.COMPLETED,
        )
        self.Job.objects.create(
            source_file="/c.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.FAILED,
        )
        self.Job.objects.create(
            source_file="/d.pdf",
            tenant_id="tenant-b",
            status=self.Job.Status.COMPLETED,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_jobs = families["ocr_tenant_jobs_total"]
        samples = {
            (s.labels["tenant_id"], s.labels["status"]): s.value
            for s in tenant_jobs.samples
        }
        assert samples[("tenant-a", "completed")] == 2
        assert samples[("tenant-a", "failed")] == 1
        assert samples[("tenant-b", "completed")] == 1

    def test_empty_tenant_id_excluded_from_tenant_jobs(self):
        """Jobs with empty tenant_id do not appear in tenant metrics."""
        self.Job.objects.create(
            source_file="/a.pdf",
            tenant_id="",
            status=self.Job.Status.COMPLETED,
        )
        self.Job.objects.create(
            source_file="/b.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.COMPLETED,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_jobs = families["ocr_tenant_jobs_total"]
        tenant_ids = {s.labels["tenant_id"] for s in tenant_jobs.samples}
        assert "" not in tenant_ids
        assert "tenant-a" in tenant_ids

    def test_tenant_pages_processed(self):
        """ocr_tenant_pages_processed counts pages per tenant."""
        job_a = self.Job.objects.create(
            source_file="/a.pdf",
            tenant_id="tenant-a",
            total_pages=3,
        )
        job_b = self.Job.objects.create(
            source_file="/b.pdf",
            tenant_id="tenant-b",
            total_pages=1,
        )
        for i in range(1, 4):
            self.PageResult.objects.create(
                job=job_a, document_id="d1", page_num=i,
                status="ok", processing_time_ms=100 * i,
            )
        self.PageResult.objects.create(
            job=job_b, document_id="d2", page_num=1,
            status="ok", processing_time_ms=200,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_pages = families["ocr_tenant_pages_processed"]
        pages_by_tenant = {
            s.labels["tenant_id"]: s.value
            for s in tenant_pages.samples
        }
        assert pages_by_tenant["tenant-a"] == 3
        assert pages_by_tenant["tenant-b"] == 1

    def test_tenant_pages_excludes_empty_tenant(self):
        """Pages from jobs with empty tenant_id are excluded."""
        job_empty = self.Job.objects.create(
            source_file="/e.pdf", tenant_id="", total_pages=1,
        )
        self.PageResult.objects.create(
            job=job_empty, document_id="d9", page_num=1,
            status="ok", processing_time_ms=100,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_pages = families["ocr_tenant_pages_processed"]
        assert len(tenant_pages.samples) == 0

    def test_tenant_processing_time_avg(self):
        """ocr_tenant_processing_time_avg_ms computes per-tenant avg."""
        job_a = self.Job.objects.create(
            source_file="/a.pdf", tenant_id="tenant-a", total_pages=2,
        )
        self.PageResult.objects.create(
            job=job_a, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
        )
        self.PageResult.objects.create(
            job=job_a, document_id="d1", page_num=2,
            status="ok", processing_time_ms=300,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_avg = families["ocr_tenant_processing_time_avg_ms"]
        avg_by_tenant = {
            s.labels["tenant_id"]: s.value
            for s in tenant_avg.samples
        }
        assert avg_by_tenant["tenant-a"] == 200.0

    def test_tenant_error_rate(self):
        """ocr_tenant_error_rate computes correct per-tenant failure rate."""
        # Create recent jobs (within last hour)
        self.Job.objects.create(
            source_file="/a.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.COMPLETED,
        )
        self.Job.objects.create(
            source_file="/b.pdf",
            tenant_id="tenant-a",
            status=self.Job.Status.FAILED,
        )
        self.Job.objects.create(
            source_file="/c.pdf",
            tenant_id="tenant-b",
            status=self.Job.Status.COMPLETED,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_err = families["ocr_tenant_error_rate"]
        err_by_tenant = {
            s.labels["tenant_id"]: s.value
            for s in tenant_err.samples
        }
        # tenant-a: 1 failed / 2 total = 0.5
        assert err_by_tenant["tenant-a"] == 0.5
        # tenant-b: 0 failed / 1 total = 0.0
        assert err_by_tenant["tenant-b"] == 0.0

    def test_tenant_error_rate_excludes_empty_tenant(self):
        """Jobs with empty tenant_id do not appear in tenant error rate."""
        self.Job.objects.create(
            source_file="/e.pdf",
            tenant_id="",
            status=self.Job.Status.FAILED,
        )

        families = {f.name: f for f in self.collector.collect()}
        tenant_err = families["ocr_tenant_error_rate"]
        assert len(tenant_err.samples) == 0

    def test_multiple_tenants_produce_separate_samples(self):
        """Each tenant gets its own metric sample across all tenant metrics."""
        job_a = self.Job.objects.create(
            source_file="/a.pdf",
            tenant_id="acme",
            status=self.Job.Status.COMPLETED,
            total_pages=1,
        )
        job_b = self.Job.objects.create(
            source_file="/b.pdf",
            tenant_id="globex",
            status=self.Job.Status.COMPLETED,
            total_pages=1,
        )
        self.PageResult.objects.create(
            job=job_a, document_id="d1", page_num=1,
            status="ok", processing_time_ms=500,
        )
        self.PageResult.objects.create(
            job=job_b, document_id="d2", page_num=1,
            status="ok", processing_time_ms=1000,
        )

        families = {f.name: f for f in self.collector.collect()}

        # Check tenant_jobs_total
        tenant_jobs = families["ocr_tenant_jobs_total"]
        tenant_ids = {s.labels["tenant_id"] for s in tenant_jobs.samples}
        assert "acme" in tenant_ids
        assert "globex" in tenant_ids

        # Check tenant_pages_processed
        tenant_pages = families["ocr_tenant_pages_processed"]
        pages = {
            s.labels["tenant_id"]: s.value
            for s in tenant_pages.samples
        }
        assert pages["acme"] == 1
        assert pages["globex"] == 1

        # Check tenant_processing_time_avg_ms
        tenant_avg = families["ocr_tenant_processing_time_avg_ms"]
        avgs = {
            s.labels["tenant_id"]: s.value
            for s in tenant_avg.samples
        }
        assert avgs["acme"] == 500.0
        assert avgs["globex"] == 1000.0

    def test_no_tenant_data_produces_empty_samples(self):
        """All tenant metrics have zero samples when no tenant jobs exist."""
        # Create a job without tenant_id
        self.Job.objects.create(
            source_file="/a.pdf", tenant_id="", total_pages=0,
        )

        families = {f.name: f for f in self.collector.collect()}
        assert len(families["ocr_tenant_jobs_total"].samples) == 0
        assert len(families["ocr_tenant_pages_processed"].samples) == 0
        assert len(families["ocr_tenant_error_rate"].samples) == 0
        assert len(families["ocr_tenant_processing_time_avg_ms"].samples) == 0


# ---------------------------------------------------------------------------
# Grafana dashboard JSON structure validation
# ---------------------------------------------------------------------------
class TestGrafanaDashboardTenantPanels:
    """Validate Grafana dashboard JSON has tenant template variable and panels."""

    @pytest.fixture
    def dashboard_json(self):
        """Extract the JSON from the Grafana dashboard configmap template."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "helm", "ocr-local", "templates",
            "grafana-dashboard-configmap.yaml",
        )
        with open(dashboard_path, encoding="utf-8") as f:
            content = f.read()
        # Extract JSON between the first '{' after 'ocr-pipeline.json: |'
        # and the last matching '}' before the Helm closing tag.
        # The template has Go template escapes like {{ "{{status}}" }}
        # which we need to handle by replacing them with plain strings.
        json_marker = "ocr-pipeline.json: |"
        idx = content.index(json_marker) + len(json_marker)
        json_text = content[idx:].strip()
        # Remove trailing Helm directives
        end_marker = "{{- end }}"
        end_idx = json_text.rfind(end_marker)
        if end_idx >= 0:
            json_text = json_text[:end_idx].strip()
        # Replace Go template escapes: {{ "{{foo}}" }} -> {{foo}}
        import re
        json_text = re.sub(r'\{\{ "(\{\{[^"]*\}\})" \}\}', r'\1', json_text)
        return json.loads(json_text)

    def test_dashboard_is_valid_json(self, dashboard_json):
        """Dashboard JSON parses without errors."""
        assert isinstance(dashboard_json, dict)
        assert "panels" in dashboard_json

    def test_tenant_template_variable_exists(self, dashboard_json):
        """Templating section includes a 'tenant' variable."""
        variables = dashboard_json.get("templating", {}).get("list", [])
        tenant_vars = [v for v in variables if v.get("name") == "tenant"]
        assert len(tenant_vars) == 1, "Expected exactly one 'tenant' template variable"
        tv = tenant_vars[0]
        assert tv["type"] == "query"
        assert tv["includeAll"] is True
        assert tv["multi"] is True
        assert "ocr_tenant_jobs_total" in tv.get("query", "")

    def test_tenant_overview_row_exists(self, dashboard_json):
        """A 'Tenant Overview' row panel exists."""
        panels = dashboard_json["panels"]
        row_panels = [
            p for p in panels
            if p.get("type") == "row" and "Tenant" in p.get("title", "")
        ]
        assert len(row_panels) >= 1, "Expected 'Tenant Overview' row panel"

    def test_tenant_panels_reference_correct_metrics(self, dashboard_json):
        """Tenant panels reference the 4 tenant metric names."""
        panels = dashboard_json["panels"]
        all_exprs = []
        for p in panels:
            for target in p.get("targets", []):
                all_exprs.append(target.get("expr", ""))

        expected_metrics = [
            "ocr_tenant_jobs_total",
            "ocr_tenant_error_rate",
            "ocr_tenant_pages_processed",
            "ocr_tenant_processing_time_avg_ms",
        ]
        for metric in expected_metrics:
            found = any(metric in expr for expr in all_exprs)
            assert found, f"No panel references metric '{metric}'"

    def test_tenant_panels_use_tenant_variable(self, dashboard_json):
        """Tenant panels filter by the $tenant template variable."""
        panels = dashboard_json["panels"]
        tenant_exprs = []
        for p in panels:
            for target in p.get("targets", []):
                expr = target.get("expr", "")
                if "ocr_tenant_" in expr and "$tenant" in expr:
                    tenant_exprs.append(expr)

        assert len(tenant_exprs) >= 4, (
            f"Expected at least 4 expressions using $tenant, got {len(tenant_exprs)}"
        )

    def test_total_panel_count(self, dashboard_json):
        """Dashboard has 53 panels (50 existing + 3 new: GPU VRAM + CPU/GPU engine + tenant storage)."""
        panels = dashboard_json["panels"]
        assert len(panels) == 53, (
            f"Expected 53 panels, got {len(panels)}"
        )
