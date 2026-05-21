"""
Unit tests for scale_test.py.

These tests cover the scale testing framework without requiring
Django, Celery, or any distributed infrastructure.

Run with: python -m pytest tests/test_scale_test.py -v
"""

import json
import os

from scale_test import (  # noqa: E402
    ScaleTestResult,
    WorkerStats,
    _minimal_pdf_bytes,
    compare_runs,
    corpus_summary,
    find_latest_result,
    generate_all_reports,
    generate_corpus,
    generate_report,
    load_results,
    save_results,
    simulate_distributed,
)

# ===========================================================================
# Tests: _minimal_pdf_bytes
# ===========================================================================


class TestMinimalPdfBytes:
    def test_single_page_pdf_valid(self):
        """Verify _minimal_pdf_bytes(1) produces bytes starting with %PDF."""
        data = _minimal_pdf_bytes(1)
        assert isinstance(data, bytes)
        assert data.startswith(b"%PDF")
        assert b"%%EOF" in data

    def test_multi_page_pdf_valid(self):
        """Verify _minimal_pdf_bytes(10) produces valid PDF structure."""
        data = _minimal_pdf_bytes(10)
        assert data.startswith(b"%PDF-1.4")
        assert b"%%EOF" in data
        # Should contain page references and content streams
        assert b"/Type /Page" in data
        assert b"/Type /Pages" in data
        assert b"/Type /Catalog" in data

    def test_pdf_page_count_in_output(self):
        """Verify the /Count N matches the requested page count."""
        for n in [1, 5, 25, 50]:
            data = _minimal_pdf_bytes(n)
            count_pattern = f"/Count {n}".encode()
            assert count_pattern in data, (
                f"Expected /Count {n} in PDF output for {n} pages"
            )

    def test_zero_pages_pdf(self):
        """Edge case: 0 pages produces a PDF structure with /Count 0."""
        data = _minimal_pdf_bytes(0)
        assert isinstance(data, bytes)
        assert data.startswith(b"%PDF")
        assert b"/Count 0" in data


# ===========================================================================
# Tests: generate_corpus
# ===========================================================================


class TestGenerateCorpus:
    def test_generates_correct_number_of_files(self, tmp_path):
        """Generate 10 docs, verify 10 PDF files created."""
        corpus = generate_corpus(10, str(tmp_path), seed=42)
        assert len(corpus) == 10
        pdf_files = list(tmp_path.glob("*.pdf"))
        assert len(pdf_files) == 10

    def test_corpus_categories_distributed(self, tmp_path):
        """Verify small/medium/large distribution roughly matches weights."""
        num_docs = 100
        corpus = generate_corpus(num_docs, str(tmp_path), seed=42)
        counts = {"small": 0, "medium": 0, "large": 0}
        for doc in corpus:
            counts[doc["size_category"]] += 1

        # With 100 docs, weights are 60/25/15.
        # Allow generous margin for rounding and shuffling.
        assert counts["small"] > counts["medium"], (
            "Small should be most common category"
        )
        assert counts["medium"] > counts["large"], (
            "Medium should be more common than large"
        )
        # At least some of each category should be present
        assert counts["small"] > 0
        assert counts["medium"] > 0
        assert counts["large"] > 0

    def test_reproducible_with_seed(self, tmp_path):
        """Two calls with same seed produce same page counts."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        corpus1 = generate_corpus(20, str(dir1), seed=123)
        corpus2 = generate_corpus(20, str(dir2), seed=123)

        pages1 = [d["pages"] for d in corpus1]
        pages2 = [d["pages"] for d in corpus2]
        assert pages1 == pages2

    def test_corpus_files_are_valid_pdfs(self, tmp_path):
        """Each generated file starts with %PDF."""
        corpus = generate_corpus(5, str(tmp_path), seed=99)
        for doc in corpus:
            with open(doc["path"], "rb") as f:
                header = f.read(5)
            assert header.startswith(b"%PDF"), (
                f"File {doc['path']} does not start with %PDF"
            )

    def test_corpus_summary_totals(self, tmp_path):
        """corpus_summary() returns correct totals."""
        corpus = generate_corpus(15, str(tmp_path), seed=7)
        summary = corpus_summary(corpus)
        assert summary["total_documents"] == 15
        expected_pages = sum(d["pages"] for d in corpus)
        assert summary["total_pages"] == expected_pages
        assert abs(summary["avg_pages_per_doc"] - expected_pages / 15) < 0.01


# ===========================================================================
# Tests: ScaleTestResult
# ===========================================================================


class TestScaleTestResult:
    def test_to_dict_excludes_job_durations(self):
        """Verify job_durations not in serialized output."""
        r = ScaleTestResult(
            test_id="abc123",
            timestamp="2026-02-14T00:00:00Z",
            mode="simulate",
            pipeline_version="0.5.0",
            config={},
            job_durations=[1.0, 2.0, 3.0],
        )
        d = r.to_dict()
        assert "job_durations" not in d
        assert d["test_id"] == "abc123"
        assert d["mode"] == "simulate"

    def test_round_trip_save_load(self, tmp_path):
        """save_results + load_results produces equivalent data."""
        r = ScaleTestResult(
            test_id="roundtrip",
            timestamp="2026-02-14T12:00:00Z",
            mode="simulate",
            pipeline_version="0.5.0",
            config={"num_jobs": 10},
            total_jobs=10,
            completed_jobs=9,
            failed_jobs=1,
            total_pages=50,
            pages_processed=45,
            fleet_ppm=120.5,
            avg_job_duration_seconds=3.14,
            per_worker_stats=[{"hostname": "w1", "pages_processed": 45}],
            corpus_info={"total_documents": 10},
        )
        filepath = save_results(r, results_dir=str(tmp_path))
        loaded = load_results(filepath)

        assert loaded.test_id == r.test_id
        assert loaded.mode == r.mode
        assert loaded.completed_jobs == r.completed_jobs
        assert loaded.failed_jobs == r.failed_jobs
        assert loaded.total_pages == r.total_pages
        assert loaded.pages_processed == r.pages_processed
        assert abs(loaded.fleet_ppm - r.fleet_ppm) < 0.01
        assert abs(loaded.avg_job_duration_seconds - r.avg_job_duration_seconds) < 0.01
        assert loaded.per_worker_stats == r.per_worker_stats
        assert loaded.corpus_info == r.corpus_info

    def test_default_values(self):
        """Fresh ScaleTestResult has zeros for metrics."""
        r = ScaleTestResult(
            test_id="defaults",
            timestamp="2026-02-14T00:00:00Z",
            mode="simulate",
            pipeline_version="0.5.0",
            config={},
        )
        assert r.total_jobs == 0
        assert r.completed_jobs == 0
        assert r.failed_jobs == 0
        assert r.cancelled_jobs == 0
        assert r.total_pages == 0
        assert r.pages_processed == 0
        assert r.pages_failed == 0
        assert r.duration_seconds == 0.0
        assert r.fleet_ppm == 0.0
        assert r.avg_job_duration_seconds == 0.0
        assert r.p50_job_duration_seconds == 0.0
        assert r.p95_job_duration_seconds == 0.0
        assert r.p99_job_duration_seconds == 0.0
        assert r.per_worker_stats == []
        assert r.error_summary == {}
        assert r.corpus_info == {}
        assert r.job_durations == []


# ===========================================================================
# Tests: simulate_distributed
# ===========================================================================


class TestSimulateDistributed:
    def test_simulate_returns_result(self):
        """Basic invocation returns ScaleTestResult."""
        result = simulate_distributed(num_jobs=5, seed=42)
        assert isinstance(result, ScaleTestResult)
        assert result.mode == "simulate"
        assert result.total_jobs == 5

    def test_simulate_job_counts(self):
        """completed + failed == total_jobs."""
        result = simulate_distributed(num_jobs=50, seed=42)
        assert result.completed_jobs + result.failed_jobs == result.total_jobs

    def test_simulate_pages_tracked(self):
        """pages_processed > 0 for completed jobs."""
        result = simulate_distributed(num_jobs=20, seed=42)
        if result.completed_jobs > 0:
            assert result.pages_processed > 0
        assert result.total_pages > 0

    def test_simulate_with_multiple_workers(self):
        """Per-worker stats has correct worker count."""
        num_workers = 4
        result = simulate_distributed(
            num_jobs=20, num_workers=num_workers, seed=42,
        )
        assert len(result.per_worker_stats) == num_workers
        for ws in result.per_worker_stats:
            assert "hostname" in ws
            assert "pages_processed" in ws
            assert "ppm" in ws

    def test_simulate_reproducible_with_seed(self):
        """Same seed gives same fleet_ppm."""
        r1 = simulate_distributed(num_jobs=30, num_workers=2, seed=999)
        r2 = simulate_distributed(num_jobs=30, num_workers=2, seed=999)
        assert abs(r1.fleet_ppm - r2.fleet_ppm) < 0.01
        assert r1.completed_jobs == r2.completed_jobs
        assert r1.failed_jobs == r2.failed_jobs
        assert r1.total_pages == r2.total_pages
        assert r1.pages_processed == r2.pages_processed

    def test_simulate_timing_percentiles_ordered(self):
        """p50 <= p95 <= p99 for job durations."""
        result = simulate_distributed(num_jobs=50, seed=42)
        assert result.p50_job_duration_seconds <= result.p95_job_duration_seconds
        assert result.p95_job_duration_seconds <= result.p99_job_duration_seconds

    def test_simulate_fleet_ppm_positive(self):
        """Fleet pages per minute should be positive for completed jobs."""
        result = simulate_distributed(num_jobs=10, seed=42)
        if result.completed_jobs > 0:
            assert result.fleet_ppm > 0

    def test_simulate_corpus_info_populated(self):
        """Corpus info should contain document and page counts."""
        result = simulate_distributed(num_jobs=15, seed=42)
        assert result.corpus_info["total_documents"] == 15
        assert result.corpus_info["total_pages"] == result.total_pages
        assert "fanout_count" in result.corpus_info
        assert "single_worker_count" in result.corpus_info

    def test_simulate_config_stored(self):
        """Config should be stored in the result."""
        result = simulate_distributed(num_jobs=5, num_workers=3, seed=42)
        assert result.config["num_jobs"] == 5
        assert result.config["num_workers"] == 3


# ===========================================================================
# Tests: Report generation
# ===========================================================================


class TestReportGeneration:
    def test_generate_report_contains_metrics(self):
        """Report string contains 'Fleet PPM'."""
        result = simulate_distributed(num_jobs=10, seed=42)
        report = generate_report(result)
        assert "Fleet PPM" in report
        assert result.test_id in report
        assert "simulate" in report
        assert "Job Summary" in report
        assert "Page Summary" in report
        assert "Performance" in report

    def test_generate_report_contains_worker_stats(self):
        """Report includes per-worker stats when multiple workers used."""
        result = simulate_distributed(num_jobs=10, num_workers=3, seed=42)
        report = generate_report(result)
        assert "Per-Worker Stats" in report
        assert "worker-0@host-0" in report

    def test_generate_all_reports_empty_dir(self, tmp_path):
        """Returns 'No scale test results found.' for empty dir."""
        report = generate_all_reports(results_dir=str(tmp_path))
        assert "No scale test results found" in report

    def test_generate_all_reports_nonexistent_dir(self):
        """Returns 'No results directory found.' for missing dir."""
        report = generate_all_reports(results_dir="/nonexistent/dir/xyz123")
        assert "No results directory found" in report

    def test_generate_all_reports_with_results(self, tmp_path):
        """Summary includes all saved runs."""
        r1 = simulate_distributed(num_jobs=5, seed=1)
        save_results(r1, results_dir=str(tmp_path))
        r2 = simulate_distributed(num_jobs=10, seed=2)
        save_results(r2, results_dir=str(tmp_path))

        report = generate_all_reports(results_dir=str(tmp_path))
        assert "2 runs" in report
        assert r1.test_id in report
        assert r2.test_id in report

    def test_compare_runs_format(self, tmp_path):
        """compare_runs produces valid comparison output."""
        r1 = simulate_distributed(num_jobs=20, seed=100)
        path1 = save_results(r1, results_dir=str(tmp_path))
        r2 = simulate_distributed(num_jobs=20, seed=200)
        path2 = save_results(r2, results_dir=str(tmp_path))

        comparison = compare_runs(path1, path2)
        assert "Baseline" in comparison
        assert "Current" in comparison
        assert r1.test_id in comparison
        assert r2.test_id in comparison
        assert "Fleet PPM" in comparison
        assert "Total Jobs" in comparison
        assert "Comparison" in comparison


# ===========================================================================
# Tests: corpus_summary
# ===========================================================================


class TestCorpusSummary:
    def test_summary_totals(self, tmp_path):
        """Correct document/page counts."""
        corpus = generate_corpus(12, str(tmp_path), seed=55)
        summary = corpus_summary(corpus)
        assert summary["total_documents"] == 12
        assert summary["total_pages"] == sum(d["pages"] for d in corpus)
        expected_avg = summary["total_pages"] / 12
        assert abs(summary["avg_pages_per_doc"] - expected_avg) < 0.01

    def test_summary_by_category(self, tmp_path):
        """Categories present in output."""
        corpus = generate_corpus(50, str(tmp_path), seed=77)
        summary = corpus_summary(corpus)
        by_cat = summary["by_category"]

        # All categories in corpus should appear in the summary
        categories_in_corpus = {d["size_category"] for d in corpus}
        for cat in categories_in_corpus:
            assert cat in by_cat
            assert by_cat[cat]["count"] > 0
            assert by_cat[cat]["pages"] > 0

        # Sum of category counts equals total documents
        total_from_cats = sum(v["count"] for v in by_cat.values())
        assert total_from_cats == 50

        # Sum of category pages equals total pages
        total_pages_from_cats = sum(v["pages"] for v in by_cat.values())
        assert total_pages_from_cats == summary["total_pages"]

    def test_summary_empty_corpus(self):
        """Empty corpus returns zero totals without error."""
        summary = corpus_summary([])
        assert summary["total_documents"] == 0
        assert summary["total_pages"] == 0
        assert summary["avg_pages_per_doc"] == 0
        assert summary["by_category"] == {}


# ===========================================================================
# Tests: Results persistence
# ===========================================================================


class TestResultsPersistence:
    def test_save_creates_file(self, tmp_path):
        """save_results creates a JSON file."""
        r = simulate_distributed(num_jobs=5, seed=42)
        filepath = save_results(r, results_dir=str(tmp_path))
        assert os.path.isfile(filepath)
        assert filepath.endswith(".json")

    def test_save_produces_valid_json(self, tmp_path):
        """Saved file contains valid JSON matching the result."""
        r = simulate_distributed(num_jobs=5, seed=42)
        filepath = save_results(r, results_dir=str(tmp_path))
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        assert data["test_id"] == r.test_id
        assert data["mode"] == "simulate"
        assert "job_durations" not in data

    def test_save_creates_results_dir(self, tmp_path):
        """save_results creates nested directories as needed."""
        r = simulate_distributed(num_jobs=3, seed=42)
        nested = str(tmp_path / "nested" / "dir")
        filepath = save_results(r, results_dir=nested)
        assert os.path.isfile(filepath)

    def test_find_latest_result(self, tmp_path):
        """find_latest_result returns the most recently saved file."""
        import time

        r1 = simulate_distributed(num_jobs=5, seed=1)
        save_results(r1, results_dir=str(tmp_path))
        # Ensure distinct mtime on Windows
        time.sleep(0.05)
        r2 = simulate_distributed(num_jobs=10, seed=2)
        path2 = save_results(r2, results_dir=str(tmp_path))
        latest = find_latest_result(str(tmp_path))
        assert latest == path2

    def test_find_latest_empty_dir(self, tmp_path):
        """Empty directory returns None."""
        assert find_latest_result(str(tmp_path)) is None

    def test_find_latest_nonexistent_dir(self):
        """Non-existent directory returns None."""
        assert find_latest_result("/nonexistent/path/xyz") is None

    def test_load_results_ignores_unknown_keys(self, tmp_path):
        """load_results ignores extra keys from future versions."""
        r = simulate_distributed(num_jobs=5, seed=42)
        filepath = save_results(r, results_dir=str(tmp_path))

        # Add an unknown key to the JSON
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        data["future_field"] = "some_value"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = load_results(filepath)
        assert loaded.test_id == r.test_id
        assert not hasattr(loaded, "future_field")


# ===========================================================================
# Tests: WorkerStats
# ===========================================================================


class TestWorkerStats:
    def test_to_dict(self):
        """WorkerStats.to_dict() returns a complete dict."""
        ws = WorkerStats(
            hostname="worker-0@host-0",
            pages_processed=100,
            tasks_completed=100,
            tasks_failed=2,
            avg_page_time_ms=350.5,
            ppm=120.0,
        )
        d = ws.to_dict()
        assert d["hostname"] == "worker-0@host-0"
        assert d["pages_processed"] == 100
        assert d["tasks_completed"] == 100
        assert d["tasks_failed"] == 2
        assert abs(d["avg_page_time_ms"] - 350.5) < 0.01
        assert abs(d["ppm"] - 120.0) < 0.01

    def test_default_values(self):
        """WorkerStats defaults to zeros."""
        ws = WorkerStats(hostname="test")
        assert ws.pages_processed == 0
        assert ws.tasks_completed == 0
        assert ws.tasks_failed == 0
        assert ws.avg_page_time_ms == 0.0
        assert ws.ppm == 0.0
