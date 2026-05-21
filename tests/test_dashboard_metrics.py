"""Tests for dashboard monitoring enhancements: p95/p99 metrics, queue depth,
engine breakdown, and DPI escalation tracking.

These tests cover the new metric families added to PipelineCollector
and the _percentile helper function.  The Django-dependent tests run
against the coordinator test database; the _percentile tests are pure
Python and always run (via mocked Django imports).

Run with:
  cd coordinator && python -m pytest jobs/tests/test_prometheus.py -v
  -- or for just the new tests from project root (with DJANGO env) --
  DJANGO_SETTINGS_MODULE=coordinator.settings_test python -m pytest tests/test_dashboard_metrics.py -v
"""

import importlib.util
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Add coordinator to sys.path for model imports, but do NOT set
# DJANGO_SETTINGS_MODULE at module level -- that would poison the env
# for all other tests in the root suite.  See .
# ---------------------------------------------------------------------------
_coordinator_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coordinator"
)
if _coordinator_dir not in sys.path:
    sys.path.insert(0, _coordinator_dir)

# Check if Django is importable AND settings are already configured
# (either via pytest-django or by the user setting DJANGO_SETTINGS_MODULE).
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


def _load_percentile_no_django():
    """Load _percentile from prometheus_metrics without Django.

    The _percentile function is pure Python (uses only math stdlib), but
    its host module ``jobs.prometheus_metrics`` has Django imports at
    module level.  This helper mocks Django and prometheus_client just
    enough to let the module load, then returns the _percentile callable.
    """
    # If Django is available, use the real import path
    if _HAS_DJANGO:
        from jobs.prometheus_metrics import _percentile
        return _percentile

    # Otherwise, mock the heavy dependencies and load via importlib
    saved = {}
    mocks = {}
    for mod_name in [
        "django", "django.utils", "django.utils.timezone", "django.conf",
        "prometheus_client", "prometheus_client.core",
        "jobs", "jobs.metrics_cache", "jobs.prometheus_metrics",
    ]:
        saved[mod_name] = sys.modules.get(mod_name)
        mocks[mod_name] = types.ModuleType(mod_name)

    try:
        # Set up Django mocks
        sys.modules["django"] = mocks["django"]
        sys.modules["django.utils"] = mocks["django.utils"]
        sys.modules["django.utils.timezone"] = mocks["django.utils.timezone"]
        sys.modules["django.conf"] = mocks["django.conf"]

        # Set up prometheus_client mocks
        prom = mocks["prometheus_client"]

        class _FakeRegistry:
            def __init__(self, **kw):
                pass

            def register(self, collector):
                pass

        prom.CollectorRegistry = _FakeRegistry
        prom.generate_latest = lambda *a: b""
        sys.modules["prometheus_client"] = prom

        prom_core = mocks["prometheus_client.core"]
        for name in [
            "CounterMetricFamily", "GaugeMetricFamily", "HistogramMetricFamily",
        ]:
            setattr(
                prom_core, name,
                type(name, (), {"__init__": lambda *a, **k: None}),
            )
        sys.modules["prometheus_client.core"] = prom_core

        # Set up jobs package mock
        jobs_pkg = mocks["jobs"]
        jobs_pkg.__path__ = [os.path.join(_coordinator_dir, "jobs")]
        jobs_pkg.__package__ = "jobs"
        sys.modules["jobs"] = jobs_pkg

        mc_mod = mocks["jobs.metrics_cache"]

        class _MockCache:
            def invalidate(self):
                pass

            def get(self, key, ttl=None, loader=None):
                if loader:
                    return loader()
                return None

            def set(self, *a, **k):
                pass

        mc_mod._cache = _MockCache()
        sys.modules["jobs.metrics_cache"] = mc_mod

        # Load the module from file
        pm_path = os.path.join(_coordinator_dir, "jobs", "prometheus_metrics.py")
        spec = importlib.util.spec_from_file_location(
            "jobs.prometheus_metrics", pm_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["jobs.prometheus_metrics"] = mod
        spec.loader.exec_module(mod)
        return mod._percentile
    finally:
        # Restore original sys.modules state
        for mod_name, original in saved.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original


# Cache the loaded function at module level so tests don't repeat the dance
_percentile_func = _load_percentile_no_django()


# ---------------------------------------------------------------------------
# Tests for _percentile helper (pure Python logic, loaded without Django)
# ---------------------------------------------------------------------------
class TestPercentileHelper:
    """Unit tests for the _percentile pure function."""

    def test_empty_list_returns_zero(self):
        assert _percentile_func([], 95) == 0.0
        assert _percentile_func([], 99) == 0.0

    def test_single_value(self):
        assert _percentile_func([42], 95) == 42.0
        assert _percentile_func([42], 50) == 42.0

    def test_known_p50(self):
        values = list(range(1, 101))
        result = _percentile_func(values, 50)
        assert result == 50.5

    def test_known_p95(self):
        values = list(range(1, 101))
        result = _percentile_func(values, 95)
        # rank = 0.95 * 99 = 94.05, lower=94 (val=95), upper=95 (val=96)
        # result = 95 + 0.05 * (96 - 95) = 95.05
        assert abs(result - 95.1) < 0.2

    def test_known_p99(self):
        values = list(range(1, 101))
        result = _percentile_func(values, 99)
        # rank = 0.99 * 99 = 98.01, lower=98 (val=99), upper=99 (val=100)
        # result = 99 + 0.01 * (100 - 99) = 99.01
        assert abs(result - 99.0) < 0.2

    def test_unsorted_input_produces_correct_result(self):
        values = [500, 100, 300, 200, 400]
        p50 = _percentile_func(values, 50)
        assert p50 == 300.0  # median of sorted [100,200,300,400,500]

    def test_two_values(self):
        values = [10, 20]
        p95 = _percentile_func(values, 95)
        # rank = 0.95 * 1 = 0.95, lower=0 (val=10), upper=1 (val=20)
        # result = 10 + 0.95 * 10 = 19.5
        assert p95 == 19.5

    def test_identical_values(self):
        values = [100, 100, 100, 100, 100]
        assert _percentile_func(values, 95) == 100.0
        assert _percentile_func(values, 99) == 100.0


# ---------------------------------------------------------------------------
# Tests for new PipelineCollector metric families (requires Django DB)
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestNewMetricFamiliesPresent:
    """Verify that all new metric families appear in collect() output."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_collect_yields_22_metric_families(self):
        """collect() yields 22 metric families (17 base + 4 tenant + 1 histogram)."""
        families = list(self.collector.collect())
        family_names = [f.name for f in families]
        assert len(families) == 22, f"Expected 22, got {len(families)}: {family_names}"

    def test_new_metric_names_present(self):
        """All 5 new metric families are present in output."""
        families = {f.name: f for f in self.collector.collect()}
        new_metrics = [
            "ocr_page_processing_time_p95_ms",
            "ocr_page_processing_time_p99_ms",
            "ocr_queue_depth",
            "ocr_dpi_escalation_total",
            "ocr_pages_by_engine",
        ]
        for metric_name in new_metrics:
            assert metric_name in families, (
                f"Expected metric '{metric_name}' not found"
            )

    def test_describe_yields_22_descriptors(self):
        """describe() yields 22 descriptors (17 base + 4 tenant + 1 histogram)."""
        descriptors = list(self.collector.describe())
        assert len(descriptors) == 22

    def test_new_descriptors_present(self):
        """All 5 new metric names appear in describe() output."""
        descriptor_names = [d.name for d in self.collector.describe()]
        new_names = [
            "ocr_page_processing_time_p95_ms",
            "ocr_page_processing_time_p99_ms",
            "ocr_queue_depth",
            "ocr_dpi_escalation_total",
            "ocr_pages_by_engine",
        ]
        for name in new_names:
            assert name in descriptor_names, (
                f"Expected descriptor '{name}' not found"
            )


@_skip_no_django
@pytest.mark.django_db
class TestP95P99MetricsWithData:
    """Verify p95/p99 computation from PageResult processing times."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_p95_p99_with_known_data(self):
        """p95 and p99 computed correctly from 20 pages with known times."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=20)
        # Create 20 pages with processing times 100, 200, ..., 2000
        for i in range(1, 21):
            self.PageResult.objects.create(
                job=job, document_id="d1", page_num=i,
                status="ok", processing_time_ms=i * 100,
                ocr_method="PaddleOCR",
            )

        families = {f.name: f for f in self.collector.collect()}

        p95_val = families["ocr_page_processing_time_p95_ms"].samples[0].value
        p99_val = families["ocr_page_processing_time_p99_ms"].samples[0].value

        # With values [100..2000], n=20:
        # p95: rank = 0.95 * 19 = 18.05
        #   lower=18 (val=1900), upper=19 (val=2000)
        #   result = 1900 + 0.05 * 100 = 1905.0
        assert abs(p95_val - 1905.0) < 1.0, f"p95={p95_val}, expected ~1905"

        # p99: rank = 0.99 * 19 = 18.81
        #   lower=18 (val=1900), upper=19 (val=2000)
        #   result = 1900 + 0.81 * 100 = 1981.0
        assert abs(p99_val - 1981.0) < 1.0, f"p99={p99_val}, expected ~1981"

    def test_p95_p99_empty_database(self):
        """p95 and p99 are 0.0 when no pages exist."""
        families = {f.name: f for f in self.collector.collect()}
        assert families["ocr_page_processing_time_p95_ms"].samples[0].value == 0.0
        assert families["ocr_page_processing_time_p99_ms"].samples[0].value == 0.0

    def test_p95_p99_excludes_zero_processing_time(self):
        """Pages with processing_time_ms=0 are excluded from percentiles."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=5)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=0,
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=500,
        )
        families = {f.name: f for f in self.collector.collect()}
        # Only one non-zero value (500), so p95 = p99 = 500
        assert families["ocr_page_processing_time_p95_ms"].samples[0].value == 500.0
        assert families["ocr_page_processing_time_p99_ms"].samples[0].value == 500.0


@_skip_no_django
@pytest.mark.django_db
class TestQueueDepthMetrics:
    """Verify queue depth metric labels and values."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job

    def test_queue_depth_labels(self):
        """Queue depth metric has expected queue labels (M-11 expansion)."""
        families = {f.name: f for f in self.collector.collect()}
        queue_family = families["ocr_queue_depth"]
        labels = {s.labels["queue"] for s in queue_family.samples}
        assert "extraction" in labels
        assert "ocr_gpu" in labels
        assert "ocr_cpu" in labels
        assert "compression" in labels
        assert "nlp" in labels
        assert "assembly" in labels

    def test_queue_depth_reflects_job_status(self):
        """Queue depth values reflect jobs in corresponding statuses."""
        self.Job.objects.create(
            source_file="/a.pdf", status=self.Job.Status.SUBMITTED,
        )
        self.Job.objects.create(
            source_file="/b.pdf", status=self.Job.Status.SUBMITTED,
        )
        self.Job.objects.create(
            source_file="/c.pdf", status=self.Job.Status.INGESTING,
        )
        self.Job.objects.create(
            source_file="/d.pdf", status=self.Job.Status.ASSEMBLING,
        )
        self.Job.objects.create(
            source_file="/e.pdf", status=self.Job.Status.COMPLETED,
        )

        families = {f.name: f for f in self.collector.collect()}
        queue_family = families["ocr_queue_depth"]
        depths = {
            s.labels["queue"]: s.value for s in queue_family.samples
        }
        assert depths["extraction"] == 1
        assert depths["assembly"] == 1
        assert depths["compression"] == 0

    def test_queue_depth_empty_database(self):
        """All queue depths are zero with empty database."""
        families = {f.name: f for f in self.collector.collect()}
        queue_family = families["ocr_queue_depth"]
        for sample in queue_family.samples:
            assert sample.value == 0


@_skip_no_django
@pytest.mark.django_db
class TestDPIEscalationMetric:
    """Verify DPI escalation count metric."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_dpi_escalation_counts_fallback_pages(self):
        """DPI escalation count matches pages with fallback status."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=5)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="fallback", processing_time_ms=200,
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="fallback", processing_time_ms=300,
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="image_only", processing_time_ms=400,
        )

        families = {f.name: f for f in self.collector.collect()}
        dpi_val = families["ocr_dpi_escalation_total"].samples[0].value
        assert dpi_val == 2

    def test_dpi_escalation_zero_when_no_fallback(self):
        """DPI escalation is 0 when no fallback pages exist."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=2)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
        )
        families = {f.name: f for f in self.collector.collect()}
        assert families["ocr_dpi_escalation_total"].samples[0].value == 0

    def test_dpi_escalation_empty_database(self):
        """DPI escalation is 0 with empty database."""
        families = {f.name: f for f in self.collector.collect()}
        assert families["ocr_dpi_escalation_total"].samples[0].value == 0


@_skip_no_django
@pytest.mark.django_db
class TestEngineBreakdownMetrics:
    """Verify pages-by-engine metric labels and values."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_engine_breakdown_by_method(self):
        """Pages are counted per ocr_method value."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=5)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=200,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="fallback", processing_time_ms=300,
            ocr_method="Tesseract",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="image_only", processing_time_ms=400,
            ocr_method="ImageOnly",
        )

        families = {f.name: f for f in self.collector.collect()}
        engine_family = families["ocr_pages_by_engine"]
        engine_counts = {
            s.labels["engine"]: s.value for s in engine_family.samples
        }
        assert engine_counts["PaddleOCR"] == 2
        assert engine_counts["Tesseract"] == 1
        assert engine_counts["ImageOnly"] == 1

    def test_engine_breakdown_excludes_empty_method(self):
        """Pages with empty ocr_method are not included."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=2)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="pending", processing_time_ms=0,
            ocr_method="",
        )
        families = {f.name: f for f in self.collector.collect()}
        engine_family = families["ocr_pages_by_engine"]
        engine_labels = [s.labels["engine"] for s in engine_family.samples]
        assert "" not in engine_labels
        assert len(engine_family.samples) == 1

    def test_engine_breakdown_empty_database(self):
        """No engine samples when database is empty."""
        families = {f.name: f for f in self.collector.collect()}
        engine_family = families["ocr_pages_by_engine"]
        assert len(engine_family.samples) == 0
