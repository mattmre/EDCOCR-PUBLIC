"""Tests for the profiling module.

Covers PipelineProfiler single-stage and multi-stage profiling, memory
tracking, JSON report format, environment info capture, thread safety,
save/load round-trip, and reset behaviour.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

from profiling import (
    ContinuousProfiler,
    FlameGraphProfiler,
    PipelineProfiler,
    StageMetrics,
    _collect_environment,
    _detect_gpu_availability,
    _get_rss_mb,
)

# ---------------------------------------------------------------------------
# StageMetrics
# ---------------------------------------------------------------------------


class TestStageMetrics:
    def test_to_dict_contains_all_fields(self):
        sm = StageMetrics(
            name="test_stage",
            wall_time_seconds=1.5,
            cpu_time_seconds=0.8,
            rss_before_mb=100.0,
            rss_after_mb=120.0,
            rss_peak_mb=130.0,
            start_timestamp="2026-01-01T00:00:00+00:00",
            end_timestamp="2026-01-01T00:00:01+00:00",
        )
        d = sm.to_dict()
        assert d["name"] == "test_stage"
        assert d["wall_time_seconds"] == 1.5
        assert d["cpu_time_seconds"] == 0.8
        assert d["rss_before_mb"] == 100.0
        assert d["rss_after_mb"] == 120.0
        assert d["rss_peak_mb"] == 130.0
        assert "start_timestamp" in d
        assert "end_timestamp" in d

    def test_to_dict_rounds_values(self):
        sm = StageMetrics(
            name="rounding",
            wall_time_seconds=1.123456789,
            cpu_time_seconds=0.987654321,
            rss_before_mb=99.999,
            rss_after_mb=100.001,
            rss_peak_mb=150.456,
        )
        d = sm.to_dict()
        assert d["wall_time_seconds"] == 1.123457
        assert d["cpu_time_seconds"] == 0.987654
        assert d["rss_before_mb"] == 100.0
        assert d["rss_after_mb"] == 100.0
        assert d["rss_peak_mb"] == 150.46


# ---------------------------------------------------------------------------
# Environment and GPU detection
# ---------------------------------------------------------------------------


class TestEnvironmentCapture:
    def test_collect_environment_has_required_keys(self):
        env = _collect_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "cpu_count" in env
        assert "pipeline_version" in env
        assert "gpu" in env

    def test_gpu_availability_returns_dict(self):
        gpu = _detect_gpu_availability()
        assert isinstance(gpu, dict)
        assert "gpu_available" in gpu
        assert "backends" in gpu
        assert isinstance(gpu["backends"], list)

    def test_get_rss_mb_returns_float(self):
        rss = _get_rss_mb()
        assert isinstance(rss, float)
        assert rss >= 0.0

    def test_collect_environment_without_psutil(self):
        with patch("profiling._HAS_PSUTIL", False):
            env = _collect_environment()
            assert "python_version" in env
            # total_memory_mb should be absent when psutil is unavailable
            assert "total_memory_mb" not in env

    def test_get_rss_mb_without_psutil(self):
        with patch("profiling._HAS_PSUTIL", False):
            assert _get_rss_mb() == 0.0


# ---------------------------------------------------------------------------
# Single-stage profiling
# ---------------------------------------------------------------------------


class TestSingleStageProfiling:
    def test_stage_captures_wall_time(self):
        profiler = PipelineProfiler()
        with profiler.stage("sleep_stage"):
            time.sleep(0.05)

        report = profiler.report()
        stages = report["stages"]
        assert len(stages) == 1
        assert stages[0]["name"] == "sleep_stage"
        # Wall time should be at least 40ms (allowing for scheduling jitter)
        assert stages[0]["wall_time_seconds"] >= 0.04

    def test_stage_captures_cpu_time(self):
        profiler = PipelineProfiler()
        with profiler.stage("cpu_work"):
            # Do some CPU work to ensure process_time advances
            total = 0
            for i in range(100000):
                total += i

        report = profiler.report()
        stages = report["stages"]
        assert len(stages) == 1
        # CPU time should be positive (some real work happened)
        assert stages[0]["cpu_time_seconds"] >= 0.0

    def test_stage_has_timestamps(self):
        profiler = PipelineProfiler()
        with profiler.stage("ts_stage"):
            pass

        report = profiler.report()
        stage = report["stages"][0]
        assert stage["start_timestamp"] != ""
        assert stage["end_timestamp"] != ""

    def test_multiple_sequential_stages(self):
        profiler = PipelineProfiler()
        with profiler.stage("stage_a"):
            time.sleep(0.02)
        with profiler.stage("stage_b"):
            time.sleep(0.02)
        with profiler.stage("stage_c"):
            time.sleep(0.02)

        report = profiler.report()
        stages = report["stages"]
        assert len(stages) == 3
        names = [s["name"] for s in stages]
        assert "stage_a" in names
        assert "stage_b" in names
        assert "stage_c" in names


# ---------------------------------------------------------------------------
# Multi-stage concurrent profiling
# ---------------------------------------------------------------------------


class TestConcurrentProfiling:
    def test_overlapping_stages_from_threads(self):
        profiler = PipelineProfiler()
        barrier = threading.Barrier(3)
        errors = []

        def run_stage(name, duration):
            try:
                barrier.wait(timeout=5)
                with profiler.stage(name):
                    time.sleep(duration)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_stage, args=("extraction", 0.05)),
            threading.Thread(target=run_stage, args=("ocr", 0.05)),
            threading.Thread(target=run_stage, args=("assembly", 0.05)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Threads raised errors: {errors}"

        report = profiler.report()
        stages = report["stages"]
        assert len(stages) == 3
        names = {s["name"] for s in stages}
        assert names == {"extraction", "ocr", "assembly"}

    def test_many_threads_stress(self):
        """Verify thread safety with many concurrent stages."""
        profiler = PipelineProfiler()
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        errors = []

        def run_stage(idx):
            try:
                barrier.wait(timeout=10)
                with profiler.stage(f"thread_{idx}"):
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_stage, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        report = profiler.report()
        assert len(report["stages"]) == num_threads


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


class TestMemoryTracking:
    def test_rss_values_captured(self):
        profiler = PipelineProfiler()
        with profiler.stage("mem_stage"):
            # Allocate some memory to ensure RSS changes are detectable
            _data = [0] * 100000
            time.sleep(0.05)  # Give memory sampler time to collect

        report = profiler.report()
        stage = report["stages"][0]
        # RSS values should be non-negative (may be 0.0 if psutil unavailable)
        assert stage["rss_before_mb"] >= 0.0
        assert stage["rss_after_mb"] >= 0.0
        assert stage["rss_peak_mb"] >= 0.0

    def test_overall_peak_rss_tracked(self):
        profiler = PipelineProfiler()
        with profiler.stage("s1"):
            time.sleep(0.02)
        with profiler.stage("s2"):
            time.sleep(0.02)

        report = profiler.report()
        overall_peak = report["overall"]["peak_rss_mb"]
        assert isinstance(overall_peak, float)
        assert overall_peak >= 0.0

    def test_peak_rss_is_max_of_stages(self):
        profiler = PipelineProfiler()
        with profiler.stage("a"):
            time.sleep(0.02)
        with profiler.stage("b"):
            time.sleep(0.02)

        report = profiler.report()
        stage_peaks = [s["rss_peak_mb"] for s in report["stages"]]
        if any(p > 0 for p in stage_peaks):
            # Overall peak should be >= max stage peak
            assert report["overall"]["peak_rss_mb"] >= max(stage_peaks) - 1.0


# ---------------------------------------------------------------------------
# JSON report format
# ---------------------------------------------------------------------------


class TestReportFormat:
    def test_report_top_level_keys(self):
        profiler = PipelineProfiler()
        with profiler.stage("x"):
            pass

        report = profiler.report()
        assert "timestamp" in report
        assert "environment" in report
        assert "stages" in report
        assert "overall" in report

    def test_overall_section_keys(self):
        profiler = PipelineProfiler()
        with profiler.stage("x"):
            pass

        overall = profiler.report()["overall"]
        assert "total_wall_time_seconds" in overall
        assert "total_cpu_time_seconds" in overall
        assert "stage_wall_sum_seconds" in overall
        assert "stage_cpu_sum_seconds" in overall
        assert "peak_rss_mb" in overall
        assert "stage_count" in overall

    def test_report_json_serializable(self):
        profiler = PipelineProfiler()
        with profiler.stage("serialize_test"):
            time.sleep(0.01)

        report = profiler.report()
        # Should not raise
        serialized = json.dumps(report, default=str)
        loaded = json.loads(serialized)
        assert loaded["stages"][0]["name"] == "serialize_test"

    def test_save_writes_valid_json(self, tmp_path):
        profiler = PipelineProfiler()
        with profiler.stage("save_test"):
            pass

        out_path = profiler.save(tmp_path / "subdir" / "profile.json")
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["stages"][0]["name"] == "save_test"
        assert "environment" in data
        assert "overall" in data

    def test_save_creates_parent_directories(self, tmp_path):
        profiler = PipelineProfiler()
        with profiler.stage("dir_test"):
            pass

        deep_path = tmp_path / "a" / "b" / "c" / "report.json"
        profiler.save(deep_path)
        assert deep_path.exists()

    def test_empty_profiler_report(self):
        """A profiler with no stages should still produce a valid report."""
        profiler = PipelineProfiler()
        report = profiler.report()
        assert report["stages"] == []
        assert report["overall"]["stage_count"] == 0
        assert report["overall"]["total_wall_time_seconds"] == 0.0

    def test_overall_wall_time_covers_all_stages(self):
        profiler = PipelineProfiler()
        with profiler.stage("first"):
            time.sleep(0.03)
        time.sleep(0.02)
        with profiler.stage("second"):
            time.sleep(0.03)

        report = profiler.report()
        overall_wall = report["overall"]["total_wall_time_seconds"]
        # Should be at least the gap between first stage start and report()
        assert overall_wall >= 0.07


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_stages(self):
        profiler = PipelineProfiler()
        with profiler.stage("before_reset"):
            pass

        assert len(profiler.report()["stages"]) == 1

        profiler.reset()
        report = profiler.report()
        assert report["stages"] == []
        assert report["overall"]["stage_count"] == 0

    def test_profiler_works_after_reset(self):
        profiler = PipelineProfiler()
        with profiler.stage("first_run"):
            pass
        profiler.reset()
        with profiler.stage("second_run"):
            pass

        report = profiler.report()
        assert len(report["stages"]) == 1
        assert report["stages"][0]["name"] == "second_run"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_stage_and_report(self):
        """Calling report() while stages are being added should not crash."""
        profiler = PipelineProfiler()
        stop = threading.Event()
        errors = []

        def add_stages():
            try:
                for i in range(50):
                    with profiler.stage(f"bg_{i}"):
                        time.sleep(0.001)
                    if stop.is_set():
                        break
            except Exception as exc:
                errors.append(exc)

        def read_reports():
            try:
                for _ in range(50):
                    profiler.report()
                    time.sleep(0.001)
                    if stop.is_set():
                        break
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_stages)
        t2 = threading.Thread(target=read_reports)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        stop.set()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_saves(self, tmp_path):
        """Multiple threads saving reports simultaneously should not corrupt."""
        profiler = PipelineProfiler()
        with profiler.stage("shared"):
            pass

        errors = []

        def save_report(idx):
            try:
                profiler.save(tmp_path / f"report_{idx}.json")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=save_report, args=(i,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        for i in range(10):
            p = tmp_path / f"report_{i}.json"
            assert p.exists()
            data = json.loads(p.read_text())
            assert data["stages"][0]["name"] == "shared"


# ---------------------------------------------------------------------------
# Integration with benchmark_comparison (import path)
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    def test_profiling_flag_accepted_by_execute(self, tmp_path):
        """Verify execute_benchmark_suite accepts enable_profiling kwarg."""
        import sys

        scripts_dir = str(
            Path(__file__).resolve().parent.parent / "scripts"
        )
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from benchmark_comparison import BenchmarkSuite, execute_benchmark_suite

        # Create a minimal input dir with a supported file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").write_bytes(b"%PDF-1.4 fake")

        calls = []

        def fake_runner(doc, src, out, script):
            calls.append(doc)
            return 100.0  # 100ms

        result = execute_benchmark_suite(
            BenchmarkSuite.STANDARD,
            input_dir,
            tmp_path / "output",
            pipeline_runner=fake_runner,
            enable_profiling=True,
        )

        assert len(calls) == 1
        assert "profiling" in result.system_info
        profiling_data = result.system_info["profiling"]
        assert "stages" in profiling_data
        assert len(profiling_data["stages"]) == 1
        assert profiling_data["stages"][0]["name"] == "document_1"


# ---------------------------------------------------------------------------
# M-17: FlameGraphProfiler
# ---------------------------------------------------------------------------


class TestFlameGraphProfiler:
    def test_context_manager(self):
        with FlameGraphProfiler() as fp:
            total = sum(range(1000))
        assert total == 499500
        report = fp.generate_report()
        assert "top_functions" in report
        assert "total_time" in report
        assert "total_calls" in report

    def test_start_stop(self):
        fp = FlameGraphProfiler()
        fp.start()
        _ = [i * i for i in range(500)]
        fp.stop()
        top = fp.get_top_functions(5)
        assert isinstance(top, list)

    def test_get_top_functions_returns_tuples(self):
        with FlameGraphProfiler() as fp:
            _ = sum(range(100))
        top = fp.get_top_functions(5)
        for entry in top:
            assert len(entry) == 3
            fn, ct, cc = entry
            assert isinstance(fn, str)
            assert isinstance(ct, float)
            assert isinstance(cc, int)

    def test_dump_stats(self, tmp_path):
        with FlameGraphProfiler() as fp:
            _ = sum(range(100))
        prof_path = fp.dump_stats(tmp_path / "sub" / "test.prof")
        assert prof_path.exists()
        assert prof_path.stat().st_size > 0

    def test_generate_report_structure(self):
        with FlameGraphProfiler() as fp:
            _ = sum(range(500))
        report = fp.generate_report()
        assert isinstance(report["top_functions"], list)
        assert isinstance(report["total_time"], float)
        assert isinstance(report["total_calls"], int)
        for func_info in report["top_functions"]:
            assert "function" in func_info
            assert "cumulative_time" in func_info
            assert "call_count" in func_info

    def test_double_stop_is_safe(self):
        fp = FlameGraphProfiler()
        fp.start()
        fp.stop()
        fp.stop()  # Should not raise

    def test_report_json_serializable(self):
        with FlameGraphProfiler() as fp:
            _ = sum(range(100))
        report = fp.generate_report()
        serialized = json.dumps(report)
        loaded = json.loads(serialized)
        assert "top_functions" in loaded


# ---------------------------------------------------------------------------
# M-20: ContinuousProfiler
# ---------------------------------------------------------------------------


class TestContinuousProfiler:
    def test_no_start_without_env_var(self):
        """ContinuousProfiler.start() should be a no-op without env var."""
        cp = ContinuousProfiler(interval=0.1)
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("CONTINUOUS_PROFILING_ENABLED", None)
            cp.start()
        assert not cp._running
        cp.stop()

    def test_starts_with_env_var(self):
        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            assert cp._running
            time.sleep(0.2)
            cp.stop()
        assert not cp._running
        summary = cp.get_summary()
        assert isinstance(summary, dict)

    def test_samples_collected(self):
        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.25)
            cp.stop()
        assert len(cp._samples) >= 2

    def test_get_summary_empty(self):
        cp = ContinuousProfiler()
        assert cp.get_summary() == {}

    def test_get_summary_has_metrics(self):
        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.25)
            cp.stop()
        summary = cp.get_summary()
        if "memory_mb" in summary:
            assert "avg" in summary["memory_mb"]
            assert "max" in summary["memory_mb"]
            assert "p95" in summary["memory_mb"]

    def test_flush_to_file(self, tmp_path):
        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.2)
            cp.stop()
        filepath = cp.flush_to_file(tmp_path / "metrics.jsonl")
        assert filepath.exists()
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) >= 1
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "timestamp" in data

    def test_flush_clears_buffer(self, tmp_path):
        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.15)
            cp.stop()
        assert len(cp._samples) > 0
        cp.flush_to_file(tmp_path / "out.jsonl")
        assert len(cp._samples) == 0

    def test_register_queue(self):
        import queue

        cp = ContinuousProfiler(interval=0.05, maxlen=100)
        q = queue.Queue()
        q.put("item")
        cp.register_queue("test_q", q)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.15)
            cp.stop()
        # Samples should include queue depth
        if cp._samples:
            last = cp._samples[-1]
            assert "queue_test_q" in last

    def test_double_start_is_safe(self):
        cp = ContinuousProfiler(interval=0.05)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            cp.start()  # Should not raise or create another thread
            assert cp._running
            cp.stop()

    def test_rolling_window_maxlen(self):
        cp = ContinuousProfiler(interval=0.02, maxlen=5)
        with patch.dict(os.environ, {"CONTINUOUS_PROFILING_ENABLED": "true"}):
            cp.start()
            time.sleep(0.3)
            cp.stop()
        assert len(cp._samples) <= 5
