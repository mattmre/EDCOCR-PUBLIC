"""
Unit tests for benchmark_pipeline.py.

These tests cover the benchmarking framework without requiring
GPU, PaddleOCR, or any heavyweight dependencies.

Run with: python -m pytest tests/test_benchmark.py -v
"""
import json
import os
import queue
import tempfile
import time

from benchmark_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    BenchmarkMetrics,
    InstrumentedQueue,
    compare_runs,
    find_latest_result,
    generate_report,
    load_results,
    save_results,
    simulate_pipeline,
)

# ===========================================================================
# Tests: BenchmarkMetrics dataclass
# ===========================================================================

class TestBenchmarkMetrics:
    def test_create_with_defaults(self):
        m = BenchmarkMetrics(
            run_id="test123",
            timestamp="2026-02-10T00:00:00Z",
            mode="simulate",
            pipeline_version="0.3.0",
            config={"NUM_WORKERS": 12},
        )
        assert m.run_id == "test123"
        assert m.mode == "simulate"
        assert m.pages_processed == 0
        assert m.pages_per_minute == 0.0
        assert m.page_timings == []

    def test_create_with_values(self):
        m = BenchmarkMetrics(
            run_id="abc",
            timestamp="2026-02-10T00:00:00Z",
            mode="live",
            pipeline_version="0.3.0",
            config={},
            pages_processed=500,
            pages_per_minute=120.5,
            avg_time_per_page_ms=498.3,
            peak_memory_mb=2048.0,
        )
        assert m.pages_processed == 500
        assert m.pages_per_minute == 120.5
        assert m.peak_memory_mb == 2048.0

    def test_to_dict_excludes_page_timings(self):
        m = BenchmarkMetrics(
            run_id="test",
            timestamp="2026-02-10T00:00:00Z",
            mode="simulate",
            pipeline_version="0.3.0",
            config={},
            page_timings=[100.0, 200.0, 300.0],
        )
        d = m.to_dict()
        assert "page_timings" not in d
        assert d["run_id"] == "test"
        assert d["mode"] == "simulate"

    def test_to_dict_includes_all_metrics(self):
        m = BenchmarkMetrics(
            run_id="test",
            timestamp="2026-02-10T00:00:00Z",
            mode="simulate",
            pipeline_version="0.3.0",
            config={"NUM_WORKERS": 12},
        )
        d = m.to_dict()
        expected_keys = {
            "run_id", "timestamp", "mode", "pipeline_version", "config",
            "total_duration_seconds", "pages_processed", "pages_per_minute",
            "avg_time_per_page_ms", "p50_time_per_page_ms",
            "p95_time_per_page_ms", "p99_time_per_page_ms",
            "peak_memory_mb", "avg_memory_mb",
            "extraction_avg_ms", "ocr_avg_ms", "assembly_avg_ms",
            "compression_avg_ms",
            "extraction_queue_throughput", "ocr_queue_throughput",
            "assembly_queue_throughput",
        }
        assert expected_keys == set(d.keys())


# ===========================================================================
# Tests: simulate_pipeline
# ===========================================================================

class TestSimulatePipeline:
    def test_simulate_returns_metrics(self):
        m = simulate_pipeline(num_pages=10)
        assert isinstance(m, BenchmarkMetrics)
        assert m.mode == "simulate"
        assert m.pages_processed == 10

    def test_simulate_positive_ppm(self):
        m = simulate_pipeline(num_pages=50)
        assert m.pages_per_minute > 0

    def test_simulate_timing_percentiles_ordered(self):
        m = simulate_pipeline(num_pages=100)
        assert m.p50_time_per_page_ms <= m.p95_time_per_page_ms
        assert m.p95_time_per_page_ms <= m.p99_time_per_page_ms

    def test_simulate_stage_timings_positive(self):
        m = simulate_pipeline(num_pages=20)
        assert m.extraction_avg_ms > 0
        assert m.ocr_avg_ms > 0
        assert m.assembly_avg_ms > 0
        assert m.compression_avg_ms > 0

    def test_simulate_memory_positive(self):
        m = simulate_pipeline(num_pages=20)
        assert m.peak_memory_mb > 0
        assert m.avg_memory_mb > 0

    def test_simulate_custom_config(self):
        config = dict(DEFAULT_CONFIG)
        config["NUM_WORKERS"] = 4
        m = simulate_pipeline(num_pages=10, config=config)
        assert m.config["NUM_WORKERS"] == 4

    def test_simulate_single_page(self):
        m = simulate_pipeline(num_pages=1)
        assert m.pages_processed == 1
        assert m.pages_per_minute > 0

    def test_simulate_page_timings_populated(self):
        m = simulate_pipeline(num_pages=10)
        assert len(m.page_timings) == 10
        assert all(t > 0 for t in m.page_timings)

    def test_simulate_throughput_positive(self):
        m = simulate_pipeline(num_pages=50)
        assert m.extraction_queue_throughput > 0
        assert m.ocr_queue_throughput > 0
        assert m.assembly_queue_throughput > 0


# ===========================================================================
# Tests: save_results / load_results round-trip
# ===========================================================================

class TestResultsPersistence:
    def test_save_creates_file(self):
        m = simulate_pipeline(num_pages=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_results(m, output_dir=tmpdir)
            assert os.path.isfile(path)
            assert path.endswith(".json")

    def test_save_load_roundtrip(self):
        m = simulate_pipeline(num_pages=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_results(m, output_dir=tmpdir)
            loaded = load_results(path)
            assert loaded.run_id == m.run_id
            assert loaded.mode == m.mode
            assert loaded.pages_processed == m.pages_processed
            assert abs(loaded.pages_per_minute - m.pages_per_minute) < 0.01

    def test_save_produces_valid_json(self):
        m = simulate_pipeline(num_pages=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_results(m, output_dir=tmpdir)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["run_id"] == m.run_id
            assert "page_timings" not in data

    def test_find_latest_result(self):
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = simulate_pipeline(num_pages=5)
            save_results(m1, output_dir=tmpdir)
            # Ensure distinct mtime on Windows (filesystem timestamp granularity)
            time.sleep(0.05)
            m2 = simulate_pipeline(num_pages=10)
            path2 = save_results(m2, output_dir=tmpdir)
            latest = find_latest_result(tmpdir)
            assert latest == path2

    def test_find_latest_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_latest_result(tmpdir) is None

    def test_find_latest_nonexistent_dir(self):
        assert find_latest_result("/nonexistent/path/xyz") is None

    def test_save_creates_output_dir(self):
        m = simulate_pipeline(num_pages=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            path = save_results(m, output_dir=nested)
            assert os.path.isfile(path)


# ===========================================================================
# Tests: generate_report
# ===========================================================================

class TestGenerateReport:
    def test_report_no_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_report(tmpdir)
            assert "No benchmark results" in report

    def test_report_nonexistent_dir(self):
        report = generate_report("/nonexistent/path/xyz")
        assert "No benchmark results" in report

    def test_report_with_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = simulate_pipeline(num_pages=20)
            save_results(m, output_dir=tmpdir)
            report = generate_report(tmpdir)
            assert m.run_id in report
            assert "Pages/Minute" in report
            assert "simulate" in report

    def test_report_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = simulate_pipeline(num_pages=10)
            save_results(m1, output_dir=tmpdir)
            m2 = simulate_pipeline(num_pages=20)
            save_results(m2, output_dir=tmpdir)
            report = generate_report(tmpdir)
            assert m1.run_id in report
            assert m2.run_id in report


# ===========================================================================
# Tests: compare_runs
# ===========================================================================

class TestCompareRuns:
    def test_compare_two_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = simulate_pipeline(num_pages=50)
            path1 = save_results(m1, output_dir=tmpdir)
            m2 = simulate_pipeline(num_pages=50)
            path2 = save_results(m2, output_dir=tmpdir)
            comparison = compare_runs(path1, path2)
            assert "Baseline" in comparison
            assert "Comparison" in comparison
            assert m1.run_id in comparison
            assert m2.run_id in comparison

    def test_compare_shows_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = simulate_pipeline(num_pages=50)
            path1 = save_results(m1, output_dir=tmpdir)
            m2 = simulate_pipeline(num_pages=50)
            path2 = save_results(m2, output_dir=tmpdir)
            comparison = compare_runs(path1, path2)
            assert "Pages/Minute" in comparison
            assert "Avg ms/page" in comparison
            assert "Peak Memory" in comparison

    def test_compare_shows_performance_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = simulate_pipeline(num_pages=50)
            path1 = save_results(m1, output_dir=tmpdir)
            m2 = simulate_pipeline(num_pages=50)
            path2 = save_results(m2, output_dir=tmpdir)
            comparison = compare_runs(path1, path2)
            assert "Performance Targets" in comparison
            assert "Slowdown" in comparison


# ===========================================================================
# Tests: InstrumentedQueue
# ===========================================================================

class TestInstrumentedQueue:
    def test_put_get_passthrough(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("hello")
        assert iq.get(timeout=1) == "hello"

    def test_records_put_times(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        iq.put("b")
        assert len(iq.put_times) == 2

    def test_records_get_times(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        iq.get(timeout=1)
        assert len(iq.get_times) == 1

    def test_records_wait_times(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        iq.get(timeout=1)
        assert len(iq.wait_times) == 1
        assert iq.wait_times[0] >= 0

    def test_qsize_delegation(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        assert iq.qsize() == 0
        iq.put("a")
        assert iq.qsize() == 1

    def test_empty_delegation(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        assert iq.empty()
        iq.put("x")
        assert not iq.empty()

    def test_task_done_delegation(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        iq.get(timeout=1)
        iq.task_done()  # Should not raise

    def test_throughput_zero_when_empty(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        assert iq.throughput() == 0.0

    def test_throughput_positive_after_gets(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        for i in range(5):
            iq.put(i)
        for _ in range(5):
            iq.get(timeout=1)
            time.sleep(0.001)  # small delay so timestamps differ
        assert iq.throughput() > 0

    def test_avg_wait_ms_zero_when_empty(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        assert iq.avg_wait_ms() == 0.0

    def test_avg_wait_ms_positive(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        iq.get(timeout=1)
        assert iq.avg_wait_ms() >= 0

    def test_avg_transit_ms_zero_when_empty(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        assert iq.avg_transit_ms() == 0.0

    def test_avg_transit_ms_positive(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put("a")
        time.sleep(0.005)
        iq.get(timeout=1)
        assert iq.avg_transit_ms() > 0

    def test_maxsize_property(self):
        iq = InstrumentedQueue(queue.Queue(maxsize=42), "test")
        assert iq.maxsize == 42

    def test_put_nowait_get_nowait(self):
        iq = InstrumentedQueue(queue.Queue(), "test")
        iq.put_nowait("x")
        assert iq.get_nowait() == "x"


# ===========================================================================
# Tests: instrument_live_pipeline
# ===========================================================================

class TestInstrumentLivePipeline:
    def test_raises_on_missing_dir(self):
        import pytest

        from benchmark_pipeline import instrument_live_pipeline  # noqa: E402

        with pytest.raises(FileNotFoundError):
            instrument_live_pipeline("/nonexistent/dir/xyz123")

    def test_empty_dir_returns_metrics(self):
        import pytest

        from benchmark_pipeline import instrument_live_pipeline  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not crash — just return empty metrics
            try:
                m = instrument_live_pipeline(tmpdir)
                assert isinstance(m, BenchmarkMetrics)
                assert m.pages_processed == 0
            except ImportError:
                pytest.skip("PaddleOCR not available")
