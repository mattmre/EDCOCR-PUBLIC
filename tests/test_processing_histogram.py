"""Tests for the processing duration histogram Prometheus metric.

Validates that ``PipelineCollector`` yields an ``ocr_processing_duration_seconds``
histogram with correct bucket boundaries, label dimensions, sum, count, and
ms-to-seconds conversion.

The ``TestDurationBucketsConstant`` tests validate a plain tuple constant and
run without Django (via mocked imports).  All other test classes require Django.

Run with:
  DJANGO_SETTINGS_MODULE=coordinator.settings_test python -m pytest tests/test_processing_histogram.py -v
"""

import importlib.util
import os
import sys
import types

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


def _load_duration_buckets_no_django():
    """Load _DURATION_BUCKETS from prometheus_metrics without Django.

    The constant is a plain tuple, but its host module has Django imports
    at module level.  This helper mocks Django and prometheus_client just
    enough to let the module load, then returns the constant.
    """
    if _HAS_DJANGO:
        from jobs.prometheus_metrics import _DURATION_BUCKETS
        return _DURATION_BUCKETS

    saved = {}
    mocks_list = [
        "django", "django.utils", "django.utils.timezone", "django.conf",
        "prometheus_client", "prometheus_client.core",
        "jobs", "jobs.metrics_cache", "jobs.prometheus_metrics",
    ]
    for mod_name in mocks_list:
        saved[mod_name] = sys.modules.get(mod_name)

    try:
        for mod_name in [
            "django", "django.utils", "django.utils.timezone", "django.conf",
        ]:
            sys.modules[mod_name] = types.ModuleType(mod_name)

        prom = types.ModuleType("prometheus_client")

        class _FakeRegistry:
            def __init__(self, **kw):
                pass

            def register(self, collector):
                pass

        prom.CollectorRegistry = _FakeRegistry
        prom.generate_latest = lambda *a: b""
        sys.modules["prometheus_client"] = prom

        prom_core = types.ModuleType("prometheus_client.core")
        for name in [
            "CounterMetricFamily", "GaugeMetricFamily", "HistogramMetricFamily",
        ]:
            setattr(
                prom_core, name,
                type(name, (), {"__init__": lambda *a, **k: None}),
            )
        sys.modules["prometheus_client.core"] = prom_core

        jobs_pkg = types.ModuleType("jobs")
        jobs_pkg.__path__ = [os.path.join(_coordinator_dir, "jobs")]
        jobs_pkg.__package__ = "jobs"
        sys.modules["jobs"] = jobs_pkg

        mc_mod = types.ModuleType("jobs.metrics_cache")

        class _MockCache:
            def invalidate(self):
                pass

            def get(self, key, ttl=None, loader=None):
                return loader() if loader else None

            def set(self, *a, **k):
                pass

        mc_mod._cache = _MockCache()
        sys.modules["jobs.metrics_cache"] = mc_mod

        pm_path = os.path.join(_coordinator_dir, "jobs", "prometheus_metrics.py")
        spec = importlib.util.spec_from_file_location(
            "jobs.prometheus_metrics", pm_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["jobs.prometheus_metrics"] = mod
        spec.loader.exec_module(mod)
        return mod._DURATION_BUCKETS
    finally:
        for mod_name, original in saved.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original


_duration_buckets = _load_duration_buckets_no_django()


# ---------------------------------------------------------------------------
# Helper: extract histogram family from collector output
# ---------------------------------------------------------------------------
def _collect_histogram(collector):
    """Return the ocr_processing_duration_seconds family from collect()."""
    for family in collector.collect():
        if family.name == "ocr_processing_duration_seconds":
            return family
    return None


# ---------------------------------------------------------------------------
# Pure-Python tests for _DURATION_BUCKETS constant (no Django needed)
# ---------------------------------------------------------------------------
class TestDurationBucketsConstant:
    """Validate the histogram bucket boundaries."""

    def test_bucket_values(self):
        assert _duration_buckets == (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)

    def test_buckets_are_sorted(self):
        assert list(_duration_buckets) == sorted(_duration_buckets)

    def test_bucket_count(self):
        assert len(_duration_buckets) == 10


# ---------------------------------------------------------------------------
# Django DB tests for histogram metric output
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestHistogramMetricPresent:
    """Verify that the histogram metric family is yielded by collect()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_histogram_family_in_collect_output(self):
        hist = _collect_histogram(self.collector)
        assert hist is not None, (
            "ocr_processing_duration_seconds not found in collect() output"
        )

    def test_histogram_in_describe(self):
        names = [d.name for d in self.collector.describe()]
        assert "ocr_processing_duration_seconds" in names

    def test_collect_yields_22_metric_families(self):
        """collect() yields 22 metric families (17 existing + 4 tenant + 1 histogram)."""
        families = list(self.collector.collect())
        family_names = [f.name for f in families]
        assert len(families) == 22, (
            f"Expected 22, got {len(families)}: {family_names}"
        )

    def test_describe_yields_22_descriptors(self):
        """describe() yields 22 descriptors (17 existing + 4 tenant + 1 histogram)."""
        descriptors = list(self.collector.describe())
        assert len(descriptors) == 22


@_skip_no_django
@pytest.mark.django_db
class TestHistogramEmptyDatabase:
    """Histogram with no PageResults yields an empty histogram family."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_empty_histogram_has_no_samples(self):
        hist = _collect_histogram(self.collector)
        assert hist is not None
        # No label combinations means no bucket/count/sum samples
        assert len(hist.samples) == 0


@_skip_no_django
@pytest.mark.django_db
class TestHistogramWithSingleEngine:
    """Histogram with pages from one engine produces correct buckets."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_single_group_bucket_counts(self):
        """Three pages at 500ms, 1500ms, 5000ms produce correct bucket counts."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=3)
        for i, ms in enumerate([500, 1500, 5000], start=1):
            self.PageResult.objects.create(
                job=job, document_id="d1", page_num=i,
                status="ok", processing_time_ms=ms,
                ocr_method="PaddleOCR",
            )

        hist = _collect_histogram(self.collector)
        assert hist is not None

        # Extract bucket samples for the (ok, PaddleOCR) label combo
        bucket_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_bucket"
            and s.labels.get("status") == "ok"
            and s.labels.get("engine") == "PaddleOCR"
        ]

        # There should be 11 buckets (10 defined + +Inf)
        assert len(bucket_samples) == 11

        # Build dict of le -> count
        le_counts = {s.labels["le"]: s.value for s in bucket_samples}

        # 500ms = 0.5s: le=0.5 should have 1
        assert le_counts["0.5"] == 1
        # 1500ms = 1.5s: le=1 should have 1, le=2 should have 2
        assert le_counts["1"] == 1
        assert le_counts["2"] == 2
        # 5000ms = 5.0s: le=5 should have 3
        assert le_counts["5"] == 3
        # All higher buckets should also be 3
        assert le_counts["10"] == 3
        assert le_counts["30"] == 3
        assert le_counts["+Inf"] == 3

    def test_sum_and_count(self):
        """Sum and count samples are correct."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=3)
        for i, ms in enumerate([500, 1500, 5000], start=1):
            self.PageResult.objects.create(
                job=job, document_id="d1", page_num=i,
                status="ok", processing_time_ms=ms,
                ocr_method="PaddleOCR",
            )

        hist = _collect_histogram(self.collector)

        count_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_count"
            and s.labels.get("status") == "ok"
        ]
        assert len(count_samples) == 1
        assert count_samples[0].value == 3

        sum_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_sum"
            and s.labels.get("status") == "ok"
        ]
        assert len(sum_samples) == 1
        # (500 + 1500 + 5000) ms = 7000 ms = 7.0 seconds
        assert sum_samples[0].value == 7.0

    def test_ms_to_seconds_conversion(self):
        """Processing times in ms are correctly converted to seconds in the histogram."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=1)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=60000,  # 60 seconds
            ocr_method="PaddleOCR",
        )

        hist = _collect_histogram(self.collector)
        bucket_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_bucket"
        ]
        le_counts = {s.labels["le"]: s.value for s in bucket_samples}

        # 60s should be in the le=60 bucket and above
        assert le_counts["30"] == 0
        assert le_counts["60"] == 1
        assert le_counts["120"] == 1
        assert le_counts["+Inf"] == 1

        # Sum should be 60.0
        sum_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_sum"
        ]
        assert sum_samples[0].value == 60.0


@_skip_no_django
@pytest.mark.django_db
class TestHistogramMultipleGroups:
    """Histogram with multiple status/engine combinations produces separate series."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_two_engine_groups(self):
        """Pages with different ocr_methods produce separate histogram series."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=4)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=1000,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=2000,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="fallback", processing_time_ms=3000,
            ocr_method="Tesseract",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="fallback", processing_time_ms=8000,
            ocr_method="Tesseract",
        )

        hist = _collect_histogram(self.collector)

        # Extract unique label combinations from count samples
        count_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_count"
        ]
        label_combos = {
            (s.labels["status"], s.labels["engine"]) for s in count_samples
        }
        assert ("ok", "PaddleOCR") in label_combos
        assert ("fallback", "Tesseract") in label_combos
        assert len(label_combos) == 2

    def test_separate_sums_per_group(self):
        """Each status/engine group has its own independent sum."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=3)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=2000,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="fallback", processing_time_ms=10000,
            ocr_method="Tesseract",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="fallback", processing_time_ms=20000,
            ocr_method="Tesseract",
        )

        hist = _collect_histogram(self.collector)
        sum_samples = {
            (s.labels["status"], s.labels["engine"]): s.value
            for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_sum"
        }
        assert sum_samples[("ok", "PaddleOCR")] == 2.0
        assert sum_samples[("fallback", "Tesseract")] == 30.0


@_skip_no_django
@pytest.mark.django_db
class TestHistogramExcludesZeroProcessingTime:
    """Pages with processing_time_ms=0 are excluded from the histogram."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_zero_ms_excluded(self):
        """A page with 0ms processing time does not appear in the histogram."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=2)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=0,
            ocr_method="PaddleOCR",
        )
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=3000,
            ocr_method="PaddleOCR",
        )

        hist = _collect_histogram(self.collector)
        count_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_count"
        ]
        # Only 1 page should be counted (the 3000ms one, not the 0ms one)
        assert len(count_samples) == 1
        assert count_samples[0].value == 1


@_skip_no_django
@pytest.mark.django_db
class TestHistogramBucketBoundaries:
    """Verify all 10 defined bucket boundaries plus +Inf are present."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.models import Job, PageResult
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()
        self.Job = Job
        self.PageResult = PageResult

    def test_all_bucket_boundaries_present(self):
        """All 11 bucket boundaries (10 defined + +Inf) are emitted."""
        job = self.Job.objects.create(source_file="/test.pdf", total_pages=1)
        self.PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=1000,
            ocr_method="PaddleOCR",
        )

        hist = _collect_histogram(self.collector)
        bucket_samples = [
            s for s in hist.samples
            if s.name == "ocr_processing_duration_seconds_bucket"
        ]

        le_values = [s.labels["le"] for s in bucket_samples]
        expected = ["0.5", "1", "2", "5", "10", "30", "60", "120", "300", "600", "+Inf"]
        assert le_values == expected
