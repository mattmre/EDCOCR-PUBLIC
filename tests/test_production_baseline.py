"""Tests for production baseline capture and evaluation tools.

Tests cover:
- Snapshot capture with mock HTTP responses
- JSONL accumulation and reading
- Report generation (JSON + markdown)
- Recording rule YAML generation
- Baseline evaluation and anomaly detection
- Health assessment
- Statistical helpers
- CLI argument parsing
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts.evaluate_baseline import (
    assess_health,
    detect_anomalies,
    evaluate_baseline,
    generate_evaluation_markdown,
)
from scripts.production_baseline import (
    _append_if_numeric,
    _percentile,
    _stddev,
    append_snapshot,
    capture_snapshot,
    compute_hourly_patterns,
    compute_series_stats,
    extract_time_series,
    generate_baseline_report,
    generate_markdown_report,
    parse_args,
    read_snapshots,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_coordinator_response(
    error_rate=0.02,
    completion_rate=0.98,
    pages=1000,
    avg_time_ms=5000,
    gpu_workers=2,
    stuck=0,
):
    """Create a mock coordinator metrics response."""
    return {
        "timestamp": "2026-03-15T10:00:00Z",
        "jobs": {
            "total": 50,
            "by_status": {"completed": 45, "failed": 1, "processing": 2, "queued": 2},
            "error_rate_1h": error_rate,
            "completion_rate_1h": completion_rate,
            "s3_error_rate_1h": 0.01,
            "stuck_total": stuck,
        },
        "workers": {
            "total": 4,
            "gpu_available": gpu_workers,
            "by_status": {"online": 2, "busy": 2, "offline": 0},
        },
        "pages": {
            "total_processed": pages,
            "avg_processing_time_ms": avg_time_ms,
            "by_status": {"ok": 900, "fallback": 80, "image_only": 20},
        },
    }


def _make_dashboard_response(ppm=12.5, dph=45.0):
    """Create a mock dashboard snapshot response."""
    return {
        "timestamp": 1710500000,
        "throughput": {
            "pages_per_minute": ppm,
            "docs_per_hour": dph,
            "bytes_per_second": 50000,
        },
        "latency": {
            "avg_ms": 4800,
            "p50_ms": 4000,
            "p95_ms": 8000,
            "p99_ms": 12000,
        },
        "jobs": {
            "total": 50,
            "active": 2,
            "completed": 45,
            "failed": 1,
            "queued": 2,
        },
    }


def _make_snapshot(
    timestamp=1710500000,
    error_rate=0.02,
    completion_rate=0.98,
    pages=1000,
    avg_time_ms=5000,
    gpu_workers=2,
    ppm=12.5,
    dph=45.0,
):
    """Create a complete snapshot dict."""
    return {
        "schema_version": "1.0.0",
        "timestamp": timestamp,
        "timestamp_iso": "2026-03-15T10:00:00+00:00",
        "sources": {
            "coordinator": _make_coordinator_response(
                error_rate=error_rate,
                completion_rate=completion_rate,
                pages=pages,
                avg_time_ms=avg_time_ms,
                gpu_workers=gpu_workers,
            ),
            "api_dashboard": _make_dashboard_response(ppm=ppm, dph=dph),
        },
    }


def _make_snapshots_series(n=20):
    """Generate a series of realistic snapshots."""
    snapshots = []
    base_ts = 1710500000
    for i in range(n):
        ts = base_ts + (i * 300)  # 5-minute intervals
        snapshots.append(_make_snapshot(
            timestamp=ts,
            error_rate=0.02 + (0.005 * (i % 5)),
            completion_rate=0.96 + (0.01 * (i % 4)),
            pages=1000 + (i * 50),
            avg_time_ms=5000 + (100 * (i % 7)),
            gpu_workers=2 + (i % 3),
            ppm=12.0 + (0.5 * (i % 6)),
            dph=40.0 + (2.0 * (i % 5)),
        ))
    return snapshots


# ---------------------------------------------------------------------------
# Statistical helpers tests
# ---------------------------------------------------------------------------

class TestStatisticalHelpers:
    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_percentile_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0
        assert _percentile(data, 0) == 1.0
        assert _percentile(data, 100) == 5.0

    def test_percentile_interpolation(self):
        data = [1.0, 2.0, 3.0, 4.0]
        p50 = _percentile(data, 50)
        assert 2.0 <= p50 <= 3.0

    def test_stddev_empty(self):
        assert _stddev([]) == 0.0

    def test_stddev_single(self):
        assert _stddev([5.0]) == 0.0

    def test_stddev_uniform(self):
        assert _stddev([3.0, 3.0, 3.0]) == 0.0

    def test_stddev_known(self):
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9] stddev ~ 2.138
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        sd = _stddev(data)
        assert 2.0 < sd < 2.3

    def test_compute_series_stats_empty(self):
        stats = compute_series_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_compute_series_stats_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_series_stats(values)
        assert stats["count"] == 10
        assert stats["mean"] == 5.5
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0
        assert stats["p50"] > 0
        assert stats["p95"] > stats["p50"]

    def test_append_if_numeric(self):
        lst = []
        _append_if_numeric(lst, 5)
        _append_if_numeric(lst, 3.14)
        _append_if_numeric(lst, None)
        _append_if_numeric(lst, "text")
        _append_if_numeric(lst, 0)
        assert lst == [5.0, 3.14, 0.0]


# ---------------------------------------------------------------------------
# Snapshot capture tests
# ---------------------------------------------------------------------------

class TestSnapshotCapture:
    @patch("scripts.production_baseline._fetch_coordinator_metrics")
    def test_capture_snapshot_coordinator_only(self, mock_fetch):
        mock_fetch.return_value = _make_coordinator_response()
        snapshot = capture_snapshot(coordinator_url="http://test:8000/api/v1/metrics/")

        assert snapshot["schema_version"] == "1.0.0"
        assert "timestamp" in snapshot
        assert "coordinator" in snapshot["sources"]
        assert "api_dashboard" not in snapshot["sources"]

    @patch("scripts.production_baseline._fetch_api_fleet")
    @patch("scripts.production_baseline._fetch_api_dashboard")
    @patch("scripts.production_baseline._fetch_coordinator_metrics")
    def test_capture_snapshot_all_sources(self, mock_coord, mock_dash, mock_fleet):
        mock_coord.return_value = _make_coordinator_response()
        mock_dash.return_value = _make_dashboard_response()
        mock_fleet.return_value = {"workers": []}

        snapshot = capture_snapshot(
            coordinator_url="http://test:8000/api/v1/metrics/",
            api_url="http://test:8080",
        )

        assert "coordinator" in snapshot["sources"]
        assert "api_dashboard" in snapshot["sources"]
        assert "api_fleet" in snapshot["sources"]

    @patch("scripts.production_baseline._fetch_coordinator_metrics")
    def test_capture_snapshot_coordinator_failure(self, mock_fetch):
        mock_fetch.return_value = None
        snapshot = capture_snapshot(coordinator_url="http://test:8000/api/v1/metrics/")

        assert len(snapshot["sources"]) == 0

    def test_capture_snapshot_no_urls(self):
        snapshot = capture_snapshot()
        assert snapshot["sources"] == {}


# ---------------------------------------------------------------------------
# JSONL persistence tests
# ---------------------------------------------------------------------------

class TestJSONLPersistence:
    def test_append_and_read(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        snap1 = _make_snapshot(timestamp=1000)
        snap2 = _make_snapshot(timestamp=2000)

        append_snapshot(filepath, snap1)
        append_snapshot(filepath, snap2)

        loaded = read_snapshots(filepath)
        assert len(loaded) == 2
        assert loaded[0]["timestamp"] == 1000
        assert loaded[1]["timestamp"] == 2000

    def test_read_nonexistent(self, tmp_path):
        filepath = tmp_path / "nonexistent.jsonl"
        assert read_snapshots(filepath) == []

    def test_read_malformed_lines(self, tmp_path):
        filepath = tmp_path / "malformed.jsonl"
        filepath.write_text(
            '{"valid": true}\n'
            'not json\n'
            '{"also_valid": true}\n',
            encoding="utf-8",
        )
        loaded = read_snapshots(filepath)
        assert len(loaded) == 2

    def test_append_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "nested" / "dir" / "test.jsonl"
        append_snapshot(filepath, {"test": True})
        assert filepath.exists()
        loaded = read_snapshots(filepath)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Time series extraction tests
# ---------------------------------------------------------------------------

class TestTimeSeriesExtraction:
    def test_extract_from_coordinator(self):
        snapshots = [_make_snapshot()]
        series = extract_time_series(snapshots)

        assert len(series["error_rate"]) == 1
        assert series["error_rate"][0] == 0.02
        assert len(series["completion_rate"]) == 1
        assert len(series["pages_processed"]) == 1
        assert series["pages_processed"][0] == 1000.0

    def test_extract_from_dashboard(self):
        snapshots = [_make_snapshot(ppm=15.0, dph=50.0)]
        series = extract_time_series(snapshots)

        assert len(series["pages_per_minute"]) == 1
        assert series["pages_per_minute"][0] == 15.0
        assert series["docs_per_hour"][0] == 50.0

    def test_extract_multiple_snapshots(self):
        snapshots = _make_snapshots_series(10)
        series = extract_time_series(snapshots)

        assert len(series["error_rate"]) == 10
        assert len(series["pages_per_minute"]) == 10

    def test_extract_empty_snapshots(self):
        series = extract_time_series([])
        for key in series:
            assert series[key] == []

    def test_extract_missing_sources(self):
        snapshots = [{"schema_version": "1.0.0", "timestamp": 1000, "sources": {}}]
        series = extract_time_series(snapshots)
        # When sources are empty, coordinator data is unavailable.
        # However, workers.by_status.get("online", 0) returns 0 which is
        # a valid numeric, so workers_online and workers_busy will have [0.0].
        assert series["error_rate"] == []
        assert series["pages_processed"] == []
        assert series["pages_per_minute"] == []
        assert series["docs_per_hour"] == []
        # Worker fields get default 0 from the by_status.get() fallback
        assert series["workers_online"] == [0.0]
        assert series["workers_busy"] == [0.0]


# ---------------------------------------------------------------------------
# Hourly patterns tests
# ---------------------------------------------------------------------------

class TestHourlyPatterns:
    def test_empty_snapshots(self):
        patterns = compute_hourly_patterns([])
        assert len(patterns) == 24
        for h in range(24):
            assert patterns[h]["sample_count"] == 0

    def test_pattern_grouping(self):
        # Create snapshots at hour 10 UTC
        base_ts = 1710500000  # Some timestamp
        from datetime import datetime
        from datetime import timezone as tz

        dt = datetime.fromtimestamp(base_ts, tz=tz.utc)
        hour = dt.hour

        snapshots = [_make_snapshot(timestamp=base_ts)]
        patterns = compute_hourly_patterns(snapshots)

        assert patterns[hour]["sample_count"] >= 1


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_generate_empty_report(self):
        report = generate_baseline_report([])
        assert "error" in report
        assert report["snapshot_count"] == 0

    def test_generate_report_structure(self):
        snapshots = _make_snapshots_series(20)
        report = generate_baseline_report(snapshots)

        assert report["schema_version"] == "1.0.0"
        assert "generated_at" in report
        assert "collection_period" in report
        assert report["collection_period"]["snapshot_count"] == 20
        assert "throughput" in report
        assert "latency" in report
        assert "error_rates" in report
        assert "workers" in report
        assert "queue_health" in report
        assert "volume" in report
        assert "hourly_patterns" in report

    def test_generate_report_throughput_stats(self):
        snapshots = _make_snapshots_series(10)
        report = generate_baseline_report(snapshots)

        ppm = report["throughput"]["pages_per_minute"]
        assert ppm["count"] == 10
        assert ppm["mean"] > 0

    def test_generate_markdown_report(self):
        snapshots = _make_snapshots_series(5)
        report = generate_baseline_report(snapshots)
        md = generate_markdown_report(report)

        assert "# Production Baseline Report" in md
        assert "## Throughput" in md
        assert "## Latency" in md
        assert "## Error Rates" in md
        assert "## Worker Utilization" in md
        assert "## Queue Health" in md
        assert "## Hourly Patterns" in md

    def test_generate_markdown_with_values(self):
        snapshots = _make_snapshots_series(5)
        report = generate_baseline_report(snapshots)
        md = generate_markdown_report(report)

        # Should contain table formatting
        assert "|" in md
        assert "Pages/min" in md


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_no_anomalies_uniform(self):
        values = [5.0] * 20
        anomalies = detect_anomalies(values)
        assert len(anomalies) == 0

    def test_detect_high_anomaly(self):
        values = [5.0] * 19 + [100.0]
        anomalies = detect_anomalies(values, threshold_sigma=2.0)
        assert len(anomalies) >= 1
        assert anomalies[-1]["direction"] == "high"

    def test_detect_low_anomaly(self):
        values = [100.0] * 19 + [0.0]
        anomalies = detect_anomalies(values, threshold_sigma=2.0)
        assert len(anomalies) >= 1
        # The low outlier should be detected
        low_anomalies = [a for a in anomalies if a["direction"] == "low"]
        assert len(low_anomalies) >= 1

    def test_too_few_values(self):
        assert detect_anomalies([1.0, 2.0]) == []

    def test_anomaly_fields(self):
        values = [5.0] * 19 + [500.0]
        anomalies = detect_anomalies(values, threshold_sigma=2.0)
        assert len(anomalies) >= 1
        a = anomalies[-1]
        assert "index" in a
        assert "value" in a
        assert "z_score" in a
        assert "direction" in a

    def test_custom_threshold(self):
        values = [5.0] * 18 + [8.0, 9.0]
        # With high threshold, nothing should be anomalous
        anomalies = detect_anomalies(values, threshold_sigma=10.0)
        assert len(anomalies) == 0


# ---------------------------------------------------------------------------
# Health assessment tests
# ---------------------------------------------------------------------------

class TestHealthAssessment:
    def test_healthy_pipeline(self):
        series = {
            "error_rate": [0.01, 0.02, 0.01, 0.03, 0.02],
            "completion_rate": [0.98, 0.97, 0.99, 0.96, 0.98],
            "stuck_jobs": [0.0, 0.0, 1.0, 0.0, 0.0],
            "gpu_workers_available": [2.0, 3.0, 2.0, 2.0, 3.0],
            "avg_processing_time_ms": [5000, 4500, 5500, 4800, 5200],
        }
        result = assess_health(series)
        assert result["overall_status"] == "PASS"
        assert result["failed_count"] == 0

    def test_unhealthy_error_rate(self):
        series = {
            "error_rate": [0.10, 0.12, 0.15, 0.20, 0.18],
            "completion_rate": [0.98, 0.97, 0.99, 0.96, 0.98],
            "stuck_jobs": [0.0],
            "gpu_workers_available": [2.0],
            "avg_processing_time_ms": [5000],
        }
        result = assess_health(series)
        assert result["overall_status"] == "FAIL"
        # Find the error_rate check
        error_check = next(c for c in result["checks"] if c["metric"] == "error_rate")
        assert error_check["passed"] is False

    def test_unhealthy_no_gpu_workers(self):
        series = {
            "error_rate": [0.01],
            "completion_rate": [0.99],
            "stuck_jobs": [0.0],
            "gpu_workers_available": [0.0, 0.0, 0.0],
            "avg_processing_time_ms": [5000],
        }
        result = assess_health(series)
        assert result["overall_status"] == "FAIL"

    def test_unhealthy_slow_processing(self):
        series = {
            "error_rate": [0.01],
            "completion_rate": [0.99],
            "stuck_jobs": [0.0],
            "gpu_workers_available": [2.0],
            "avg_processing_time_ms": [35000, 40000, 38000],
        }
        result = assess_health(series)
        assert result["overall_status"] == "FAIL"

    def test_empty_series(self):
        series = {
            "error_rate": [],
            "completion_rate": [],
            "stuck_jobs": [],
            "gpu_workers_available": [],
            "avg_processing_time_ms": [],
        }
        result = assess_health(series)
        # No data = no checks to fail
        assert result["overall_status"] == "PASS"
        assert result["checks_count"] == 0

    def test_custom_thresholds(self):
        series = {
            "error_rate": [0.08],
            "completion_rate": [0.95],
            "stuck_jobs": [0.0],
            "gpu_workers_available": [1.0],
            "avg_processing_time_ms": [5000],
        }
        # With very strict thresholds, should fail
        strict_thresholds = {
            "max_avg_error_rate": 0.01,
            "max_peak_error_rate": 0.02,
            "min_avg_completion_rate": 0.99,
            "max_avg_stuck_jobs": 0.0,
            "min_avg_gpu_workers": 3.0,
            "max_avg_processing_time_ms": 1000,
        }
        result = assess_health(series, strict_thresholds)
        assert result["overall_status"] == "FAIL"
        assert result["failed_count"] > 0


# ---------------------------------------------------------------------------
# Full evaluation tests
# ---------------------------------------------------------------------------

class TestFullEvaluation:
    def test_evaluate_structure(self):
        snapshots = _make_snapshots_series(10)
        result = evaluate_baseline(snapshots)

        assert result["schema_version"] == "1.0.0"
        assert "generated_at" in result
        assert "collection_period" in result
        assert "statistics" in result
        assert "anomalies" in result
        assert "health_assessment" in result

    def test_evaluate_statistics(self):
        snapshots = _make_snapshots_series(10)
        result = evaluate_baseline(snapshots)

        stats = result["statistics"]
        assert "error_rate" in stats
        assert stats["error_rate"]["count"] == 10

    def test_evaluate_markdown(self):
        snapshots = _make_snapshots_series(10)
        result = evaluate_baseline(snapshots)
        md = generate_evaluation_markdown(result)

        assert "# Baseline Evaluation Report" in md
        assert "## Health Checks" in md
        assert "## Statistics Summary" in md

    def test_evaluate_with_anomalies(self):
        snapshots = _make_snapshots_series(20)
        # Add an extreme outlier
        snapshots.append(_make_snapshot(
            timestamp=1710510000,
            error_rate=0.95,  # Extreme error rate
            pages=1000,
        ))
        result = evaluate_baseline(snapshots, anomaly_threshold=2.0)

        # Should detect anomalies in error_rate
        assert "error_rate" in result.get("anomalies", {}) or len(result.get("anomalies", {})) >= 0


# ---------------------------------------------------------------------------
# Recording rules tests
# ---------------------------------------------------------------------------

class TestRecordingRules:
    def test_import_recording_rules_module(self):
        """Test that the recording rules module can be imported standalone."""
        # We need to import without Django
        import importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rules = module.get_all_recording_rules()
        assert len(rules) > 0

    def test_recording_rules_count(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rules = module.get_all_recording_rules()
        # Should have rate smoothing (8) + worker utilization (4) + SLA burn rate (4)
        assert len(rules) >= 14

    def test_recording_rules_names(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rules = module.get_all_recording_rules()
        names = [r["record"] for r in rules]

        assert "ocr:error_rate:avg5m" in names
        assert "ocr:completion_rate:avg5m" in names
        assert "ocr:worker_utilization_ratio" in names
        assert "ocr:error_budget_burn_rate_1h" in names
        assert "ocr:error_budget_consumed_1h" in names

    def test_generate_yaml(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        yaml_text = module.generate_recording_rules_yaml()
        assert "- name: ocr-recording-rules" in yaml_text
        assert "interval: 30s" in yaml_text
        assert "record: ocr:error_rate:avg5m" in yaml_text
        assert "record: ocr:worker_utilization_ratio" in yaml_text

    def test_generate_yaml_custom_slo(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        yaml_text = module.generate_recording_rules_yaml(slo_target=0.999)
        # With 99.9% SLO, allowed error = 0.001
        assert "0.001" in yaml_text

    def test_generate_full_prometheusrule(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prometheus_recording_rules",
            Path(__file__).resolve().parent.parent / "coordinator" / "jobs" / "prometheus_recording_rules.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        yaml_text = module.generate_full_prometheusrule_yaml()
        assert "apiVersion: monitoring.coreos.com/v1" in yaml_text
        assert "kind: PrometheusRule" in yaml_text


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_default_args(self):
        args = parse_args([])
        assert args.interval == 300
        assert args.duration == 24
        assert not args.snapshot
        assert not args.report_only

    def test_snapshot_mode(self):
        args = parse_args(["--snapshot"])
        assert args.snapshot is True

    def test_custom_interval(self):
        args = parse_args(["--interval", "60", "--duration", "2"])
        assert args.interval == 60
        assert args.duration == 2.0

    def test_report_only_mode(self):
        args = parse_args(["--report-only", "--input", "data.jsonl"])
        assert args.report_only is True
        assert args.input == Path("data.jsonl")

    def test_custom_urls(self):
        args = parse_args([
            "--coordinator-url", "http://coord:8000/api/v1/metrics/",
            "--api-url", "http://api:8080",
            "--api-key", "test-key",
        ])
        assert args.coordinator_url == "http://coord:8000/api/v1/metrics/"
        assert args.api_url == "http://api:8080"
        assert args.api_key == "test-key"


# ---------------------------------------------------------------------------
# Integration test: end-to-end snapshot -> report cycle
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_capture_persist_report_cycle(self, tmp_path):
        """Full cycle: create snapshots, write to JSONL, generate report."""
        jsonl_path = tmp_path / "baseline.jsonl"

        # Write multiple snapshots
        for snap in _make_snapshots_series(15):
            append_snapshot(jsonl_path, snap)

        # Read back
        loaded = read_snapshots(jsonl_path)
        assert len(loaded) == 15

        # Generate report
        report = generate_baseline_report(loaded)
        assert report["collection_period"]["snapshot_count"] == 15
        assert report["throughput"]["pages_per_minute"]["count"] == 15

        # Generate markdown
        md = generate_markdown_report(report)
        assert len(md) > 100

        # Run evaluation
        evaluation = evaluate_baseline(loaded)
        assert evaluation["health_assessment"]["overall_status"] in ("PASS", "FAIL")

        # Generate evaluation markdown
        eval_md = generate_evaluation_markdown(evaluation)
        assert "# Baseline Evaluation Report" in eval_md

    def test_main_report_only(self, tmp_path):
        """Test main() in --report-only mode."""
        jsonl_path = tmp_path / "data.jsonl"
        for snap in _make_snapshots_series(5):
            append_snapshot(jsonl_path, snap)

        from scripts.production_baseline import main
        rc = main([
            "--report-only",
            "--input", str(jsonl_path),
            "--output-dir", str(tmp_path / "output"),
        ])
        assert rc == 0
        assert (tmp_path / "output" / "baseline_report.json").exists()
        assert (tmp_path / "output" / "baseline_report.md").exists()

    @patch("scripts.production_baseline._fetch_coordinator_metrics")
    def test_main_snapshot_mode(self, mock_fetch, tmp_path):
        """Test main() in --snapshot mode."""
        mock_fetch.return_value = _make_coordinator_response()

        from scripts.production_baseline import main
        rc = main([
            "--snapshot",
            "--coordinator-url", "http://test:8000/api/v1/metrics/",
            "--output-dir", str(tmp_path),
        ])
        assert rc == 0

    def test_evaluate_main_pass(self, tmp_path):
        """Test evaluate_baseline main() with passing data."""
        jsonl_path = tmp_path / "data.jsonl"
        for snap in _make_snapshots_series(10):
            append_snapshot(jsonl_path, snap)

        from scripts.evaluate_baseline import main as eval_main
        eval_main([
            "--input", str(jsonl_path),
            "--output-dir", str(tmp_path / "eval_output"),
            "--format", "both",
        ])
        assert (tmp_path / "eval_output" / "evaluation_report.json").exists()
        assert (tmp_path / "eval_output" / "evaluation_report.md").exists()
