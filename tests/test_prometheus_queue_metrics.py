"""Tests for M-11 queue depth gauge and M-13 page throughput counter.

Covers:
- ocr_queue_depth Gauge with 'queue' label (6 pipeline stages)
- ocr_pages_processed_total Counter with 'engine' and 'status' labels
- Metric type validation (Gauge vs Counter)
- Label set correctness
- Engine name normalization (PaddleOCR -> paddle, Tesseract -> tesseract)
- Status mapping (ok/completed/fallback/image_only -> success, failed -> failed)

Run with:
  DJANGO_SETTINGS_MODULE=coordinator.settings_test python -m pytest tests/test_prometheus_queue_metrics.py -v
"""

import importlib.util
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
# M-11: Queue depth gauge tests
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestQueueDepthGaugeM11:
    """Verify ocr_queue_depth Gauge has the correct label set and stages."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_queue_depth_metric_exists(self):
        """ocr_queue_depth is present in collect() output."""
        families = {f.name: f for f in self.collector.collect()}
        assert "ocr_queue_depth" in families

    def test_queue_depth_is_gauge_type(self):
        """ocr_queue_depth is a Gauge metric family."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        assert family.type == "gauge"

    def test_queue_depth_has_queue_label(self):
        """ocr_queue_depth samples use 'queue' label (not 'queue_name')."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        for sample in family.samples:
            assert "queue" in sample.labels
            assert "queue_name" not in sample.labels

    def test_queue_depth_has_all_six_stages(self):
        """ocr_queue_depth has samples for all 6 pipeline stages."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        labels = {s.labels["queue"] for s in family.samples}
        expected = {"extraction", "ocr_gpu", "ocr_cpu", "compression", "nlp", "assembly"}
        assert labels == expected

    def test_queue_depth_empty_database_all_zero(self):
        """All queue depths are zero with empty database."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        for sample in family.samples:
            assert sample.value == 0

    def test_queue_depth_extraction_counts_ingesting(self):
        """extraction queue depth reflects INGESTING jobs."""
        from jobs.models import Job

        Job.objects.create(source_file="/a.pdf", status=Job.Status.INGESTING)
        Job.objects.create(source_file="/b.pdf", status=Job.Status.INGESTING)
        Job.objects.create(source_file="/c.pdf", status=Job.Status.COMPLETED)

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        depths = {s.labels["queue"]: s.value for s in family.samples}
        assert depths["extraction"] == 2

    def test_queue_depth_assembly_counts_assembling(self):
        """assembly queue depth reflects ASSEMBLING jobs."""
        from jobs.models import Job

        Job.objects.create(source_file="/a.pdf", status=Job.Status.ASSEMBLING)

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        depths = {s.labels["queue"]: s.value for s in family.samples}
        assert depths["assembly"] == 1

    def test_queue_depth_ocr_gpu_with_gpu_worker(self):
        """ocr_gpu queue depth counts PROCESSING jobs assigned to GPU workers."""
        from jobs.models import Job, Worker

        Worker.objects.create(
            hostname="gpu-worker-1",
            status=Worker.Status.BUSY,
            gpu_available=True,
        )
        Job.objects.create(
            source_file="/a.pdf",
            status=Job.Status.PROCESSING,
            assigned_worker="gpu-worker-1",
        )
        Job.objects.create(
            source_file="/b.pdf",
            status=Job.Status.PROCESSING,
            assigned_worker="cpu-worker-1",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        depths = {s.labels["queue"]: s.value for s in family.samples}
        assert depths["ocr_gpu"] == 1
        assert depths["ocr_cpu"] == 1

    def test_queue_depth_ocr_cpu_no_gpu_workers(self):
        """All PROCESSING jobs go to ocr_cpu when no GPU workers exist."""
        from jobs.models import Job

        Job.objects.create(
            source_file="/a.pdf",
            status=Job.Status.PROCESSING,
            assigned_worker="some-worker",
        )
        Job.objects.create(
            source_file="/b.pdf",
            status=Job.Status.PROCESSING,
            assigned_worker="other-worker",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_queue_depth"]
        depths = {s.labels["queue"]: s.value for s in family.samples}
        assert depths["ocr_gpu"] == 0
        assert depths["ocr_cpu"] == 2

    def test_queue_depth_in_describe(self):
        """ocr_queue_depth appears in describe() output."""
        descriptor_names = [d.name for d in self.collector.describe()]
        assert "ocr_queue_depth" in descriptor_names


# ---------------------------------------------------------------------------
# M-13: Page throughput counter tests
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestPageThroughputCounterM13:
    """Verify ocr_pages_processed_total Counter has engine and status labels."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_pages_processed_metric_exists(self):
        """ocr_pages_processed_total is present in collect() output."""
        families = {f.name: f for f in self.collector.collect()}
        assert "ocr_pages_processed_total" in families

    def test_pages_processed_is_counter_type(self):
        """ocr_pages_processed_total is a Counter metric family."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        assert family.type == "counter"

    def test_pages_processed_has_engine_and_status_labels(self):
        """Samples have both 'engine' and 'status' labels."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=1)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        assert len(family.samples) > 0
        for sample in family.samples:
            assert "engine" in sample.labels
            assert "status" in sample.labels

    def test_pages_processed_empty_database(self):
        """No samples when database has no pages."""
        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        assert len(family.samples) == 0

    def test_engine_normalization_paddle(self):
        """PaddleOCR ocr_method is normalized to 'paddle' engine label."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=2)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=200,
            ocr_method="PaddleOCR",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert ("paddle", "success") in engine_status
        assert engine_status[("paddle", "success")] == 2

    def test_engine_normalization_tesseract(self):
        """Tesseract ocr_method is normalized to 'tesseract' engine label."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=1)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="fallback", processing_time_ms=300,
            ocr_method="Tesseract",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert ("tesseract", "success") in engine_status
        assert engine_status[("tesseract", "success")] == 1

    def test_engine_normalization_onnx(self):
        """ONNX ocr_method variants are normalized to 'onnx' engine label."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=1)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=150,
            ocr_method="ONNX",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert ("onnx", "success") in engine_status

    def test_status_mapping_success(self):
        """ok, completed, fallback, and image_only statuses map to 'success'."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=4)
        for i, status in enumerate(["ok", "completed", "fallback", "image_only"], start=1):
            PageResult.objects.create(
                job=job, document_id="d1", page_num=i,
                status=status, processing_time_ms=100,
                ocr_method="PaddleOCR",
            )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert engine_status[("paddle", "success")] == 4

    def test_status_mapping_failed(self):
        """failed status maps to 'failed'."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=1)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="failed", processing_time_ms=0,
            ocr_method="PaddleOCR",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert ("paddle", "failed") in engine_status
        assert engine_status[("paddle", "failed")] == 1

    def test_failed_pages_no_engine_counted_as_unknown(self):
        """Failed pages with empty ocr_method get 'unknown' engine label."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=1)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="failed", processing_time_ms=0,
            ocr_method="",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert ("unknown", "failed") in engine_status
        assert engine_status[("unknown", "failed")] == 1

    def test_multiple_engines_produce_separate_samples(self):
        """Different engines produce separate counter samples."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=3)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="fallback", processing_time_ms=200,
            ocr_method="Tesseract",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="failed", processing_time_ms=0,
            ocr_method="PaddleOCR",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        engine_status = {
            (s.labels["engine"], s.labels["status"]): s.value
            for s in family.samples
        }
        assert engine_status[("paddle", "success")] == 1
        assert engine_status[("tesseract", "success")] == 1
        assert engine_status[("paddle", "failed")] == 1

    def test_pages_processed_in_describe(self):
        """ocr_pages_processed_total appears in describe() output."""
        descriptor_names = [d.name for d in self.collector.describe()]
        assert "ocr_pages_processed_total" in descriptor_names

    def test_pending_pages_excluded(self):
        """Pages with pending status are not counted in throughput."""
        from jobs.models import Job, PageResult

        job = Job.objects.create(source_file="/test.pdf", total_pages=2)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="pending", processing_time_ms=0,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", processing_time_ms=100,
            ocr_method="PaddleOCR",
        )

        families = {f.name: f for f in self.collector.collect()}
        family = families["ocr_pages_processed_total"]
        total = sum(s.value for s in family.samples)
        assert total == 1


# ---------------------------------------------------------------------------
# Combined metric family count validation
# ---------------------------------------------------------------------------
@_skip_no_django
@pytest.mark.django_db
class TestMetricFamilyCount:
    """Verify total metric family count remains consistent after M-11/M-13."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from jobs.metrics_cache import _cache
        from jobs.prometheus_metrics import PipelineCollector

        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_collect_yields_22_families(self):
        """collect() yields 22 metric families (unchanged count, types updated)."""
        families = list(self.collector.collect())
        family_names = [f.name for f in families]
        assert len(families) == 22, f"Expected 22, got {len(families)}: {family_names}"

    def test_describe_yields_22_descriptors(self):
        """describe() yields 22 descriptors (unchanged count)."""
        descriptors = list(self.collector.describe())
        assert len(descriptors) == 22
