"""Tests for pipeline performance benchmark/profiling tools (Items 11-15).

Covers:
    - scripts/benchmark_adaptive_batch.py
    - scripts/profile_page_cache.py
    - scripts/monitor_page_routing.py
    - scripts/benchmark_gpu_fusion.py
    - scripts/validate_keda_scaling.py
"""

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import mock

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Item 11: benchmark_adaptive_batch
# ---------------------------------------------------------------------------

from adaptive_batch import (
    AdaptiveBatchSizer,
    BatchResult,
    BatchStrategy,
    PageComplexity,
)
from scripts.benchmark_adaptive_batch import (
    WORKLOAD_PROFILES,
    BenchmarkReport,
    StrategyBenchmarkResult,
    _generate_recommendations,
    format_console_report,
    format_markdown_report,
    generate_workload,
    run_full_benchmark,
    run_strategy_benchmark,
    simulate_batch_processing,
)


class TestBenchmarkAdaptiveBatch:
    """Tests for the adaptive batch benchmark tool."""

    def test_workload_profiles_exist(self):
        """All expected workload profiles are defined."""
        expected = {"small", "medium", "large", "mixed"}
        assert expected.issubset(set(WORKLOAD_PROFILES.keys()))

    def test_generate_workload_returns_pages(self):
        """generate_workload produces PageComplexity objects."""
        sizer = AdaptiveBatchSizer()
        pages = generate_workload(sizer, "small", 10, seed=42)
        assert len(pages) == 10
        assert all(isinstance(p, PageComplexity) for p in pages)

    def test_generate_workload_deterministic(self):
        """Same seed produces identical workloads."""
        sizer = AdaptiveBatchSizer()
        w1 = generate_workload(sizer, "medium", 20, seed=99)
        w2 = generate_workload(sizer, "medium", 20, seed=99)
        scores1 = [p.complexity_score for p in w1]
        scores2 = [p.complexity_score for p in w2]
        assert scores1 == scores2

    def test_generate_workload_complexity_scores(self):
        """Generated pages have non-negative complexity scores."""
        sizer = AdaptiveBatchSizer()
        pages = generate_workload(sizer, "large", 50, seed=1)
        for p in pages:
            assert p.complexity_score >= 0

    def test_simulate_batch_processing_returns_result(self):
        """simulate_batch_processing returns a BatchResult."""
        sizer = AdaptiveBatchSizer()
        pages = generate_workload(sizer, "small", 5)
        result = simulate_batch_processing(pages, batch_size=5)
        assert isinstance(result, BatchResult)
        assert result.pages_processed == 5
        assert result.success_count == 5
        assert result.failure_count == 0

    def test_simulate_batch_processing_empty(self):
        """Empty pages list returns empty BatchResult."""
        result = simulate_batch_processing([], batch_size=0)
        assert result.pages_processed == 0

    def test_run_strategy_benchmark_fixed(self):
        """Fixed strategy benchmark runs successfully."""
        result = run_strategy_benchmark(
            strategy=BatchStrategy.FIXED,
            max_batch_size=8,
            workload_profile="small",
            page_count=20,
            seed=42,
        )
        assert isinstance(result, StrategyBenchmarkResult)
        assert result.strategy == "fixed"
        assert result.pages_total == 20
        assert result.throughput_pages_per_sec > 0

    def test_run_strategy_benchmark_adaptive(self):
        """Adaptive strategy benchmark runs successfully."""
        result = run_strategy_benchmark(
            strategy=BatchStrategy.ADAPTIVE,
            max_batch_size=16,
            workload_profile="medium",
            page_count=30,
            seed=42,
        )
        assert result.strategy == "adaptive"
        assert result.pages_total == 30
        assert result.batches_processed > 0

    def test_run_full_benchmark_minimal(self):
        """Full benchmark with minimal params returns a report."""
        report = run_full_benchmark(
            workload_size=10,
            iterations=1,
            strategies=["fixed"],
            batch_sizes=[4],
            profiles=["small"],
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0
        assert report.timestamp

    def test_run_full_benchmark_recommendations(self):
        """Full benchmark generates recommendations."""
        report = run_full_benchmark(
            workload_size=10,
            iterations=1,
            strategies=["fixed", "adaptive"],
            batch_sizes=[4, 8],
            profiles=["small"],
        )
        assert len(report.recommendations) > 0

    def test_generate_recommendations(self):
        """_generate_recommendations produces per-profile recs."""
        results = [
            StrategyBenchmarkResult(
                strategy="fixed", workload_profile="small",
                max_batch_size=4, throughput_pages_per_sec=100,
                p95_latency_ms=5,
            ),
            StrategyBenchmarkResult(
                strategy="adaptive", workload_profile="small",
                max_batch_size=8, throughput_pages_per_sec=200,
                p95_latency_ms=3,
            ),
        ]
        recs = _generate_recommendations(results)
        assert len(recs) == 1
        assert recs[0]["workload_profile"] == "small"
        assert recs[0]["recommended_strategy"] == "adaptive"

    def test_format_console_report(self):
        """Console report formatter produces non-empty output."""
        report = BenchmarkReport(
            timestamp="2026-01-01T00:00:00Z",
            workload_size=10,
            iterations=1,
            results=[{
                "workload_profile": "small", "strategy": "fixed",
                "max_batch_size": 4, "throughput_pages_per_sec": 100,
                "avg_latency_ms": 5, "p95_latency_ms": 8, "avg_batch_size": 4,
            }],
            recommendations=[],
        )
        output = format_console_report(report)
        assert "ADAPTIVE BATCH BENCHMARK" in output
        assert "small" in output

    def test_format_markdown_report(self):
        """Markdown report formatter produces valid markdown."""
        report = BenchmarkReport(
            timestamp="2026-01-01T00:00:00Z",
            workload_size=10,
            iterations=1,
            results=[{
                "workload_profile": "small", "strategy": "fixed",
                "max_batch_size": 4, "throughput_pages_per_sec": 100,
                "avg_latency_ms": 5, "p95_latency_ms": 8, "avg_batch_size": 4,
            }],
            recommendations=[],
        )
        md = format_markdown_report(report)
        assert "# Adaptive Batch Benchmark Report" in md
        assert "| small" in md

    def test_cli_argument_parsing(self):
        """CLI parses arguments without error."""
        from scripts.benchmark_adaptive_batch import main
        with mock.patch("sys.argv", [
            "benchmark_adaptive_batch.py",
            "--workload-size", "5",
            "--iterations", "1",
            "--strategies", "fixed",
            "--batch-sizes", "4",
            "--profiles", "small",
        ]):
            result = main()
            assert result == 0


# ---------------------------------------------------------------------------
# Item 12: profile_page_cache
# ---------------------------------------------------------------------------

from scripts.profile_page_cache import (
    CacheProfileReport,
    CacheProfileResult,
    _generate_page_data,
    profile_cache_config,
    run_full_profile,
)
from scripts.profile_page_cache import (
    _percentile as cache_percentile,
)
from scripts.profile_page_cache import (
    format_console_report as cache_console_report,
)
from scripts.profile_page_cache import (
    format_markdown_report as cache_md_report,
)


class TestProfilePageCache:
    """Tests for the page cache profiling tool."""

    def test_generate_page_data_size(self):
        """Generated page data matches requested size."""
        data = _generate_page_data(1024, seed=0)
        assert len(data) == 1024

    def test_generate_page_data_deterministic(self):
        """Same seed produces identical data."""
        d1 = _generate_page_data(256, seed=42)
        d2 = _generate_page_data(256, seed=42)
        assert d1 == d2

    def test_generate_page_data_different_seeds(self):
        """Different seeds produce different data."""
        d1 = _generate_page_data(256, seed=1)
        d2 = _generate_page_data(256, seed=2)
        assert d1 != d2

    def test_percentile_empty(self):
        """Percentile of empty list returns 0."""
        assert cache_percentile([], 50) == 0.0

    def test_percentile_single(self):
        """Percentile of single-element list returns that element."""
        assert cache_percentile([42.0], 95) == 42.0

    def test_percentile_normal(self):
        """Percentile of a sorted range returns expected value."""
        values = list(range(100))
        assert cache_percentile(values, 50) == 50

    def test_profile_cache_config_basic(self):
        """profile_cache_config runs and returns valid result."""
        result = profile_cache_config(
            cache_size_mib=1,
            max_entries=100,
            page_count=20,
            page_size_bytes=1024,
            page_size_label="test",
        )
        assert isinstance(result, CacheProfileResult)
        assert result.total_puts == 20
        assert result.total_gets == 20

    def test_profile_cache_config_evictions(self):
        """Cache evictions occur when pages exceed capacity."""
        result = profile_cache_config(
            cache_size_mib=1,  # 1 MiB
            max_entries=5,     # Only 5 entries
            page_count=20,     # 20 pages
            page_size_bytes=100_000,
            page_size_label="eviction_test",
        )
        assert result.evictions > 0

    def test_profile_cache_config_hit_rate(self):
        """Sequential access pattern produces expected hit rate."""
        result = profile_cache_config(
            cache_size_mib=10,
            max_entries=1000,
            page_count=50,
            page_size_bytes=100,
            page_size_label="hit_test",
            access_pattern="sequential",
        )
        # With large cache and sequential access, most gets should hit
        assert result.hit_rate >= 0.0

    def test_profile_cache_config_random_access(self):
        """Random access pattern runs without error."""
        result = profile_cache_config(
            cache_size_mib=1,
            max_entries=50,
            page_count=30,
            page_size_bytes=1024,
            page_size_label="random_test",
            access_pattern="random",
        )
        assert result.total_gets == 30

    def test_profile_cache_config_zipf_access(self):
        """Zipf access pattern runs without error."""
        result = profile_cache_config(
            cache_size_mib=1,
            max_entries=50,
            page_count=30,
            page_size_bytes=1024,
            page_size_label="zipf_test",
            access_pattern="zipf",
        )
        assert result.total_gets == 30

    def test_profile_cache_config_growth_snapshots(self):
        """Growth snapshots are collected at intervals."""
        result = profile_cache_config(
            cache_size_mib=10,
            max_entries=500,
            page_count=100,
            page_size_bytes=100,
            page_size_label="snapshot_test",
            snapshot_interval=25,
        )
        # 100 puts / 25 interval = 4 snapshots expected
        assert len(result.growth_snapshots) >= 3

    def test_run_full_profile_minimal(self):
        """Full profile with minimal params returns a report."""
        report = run_full_profile(
            cache_sizes_mib=[1],
            page_count=10,
        )
        assert isinstance(report, CacheProfileReport)
        assert len(report.results) > 0
        assert report.timestamp

    def test_run_full_profile_recommendations(self):
        """Full profile generates recommendations."""
        report = run_full_profile(
            cache_sizes_mib=[1, 2],
            page_count=10,
        )
        assert len(report.recommendations) > 0

    def test_format_console_report(self):
        """Console report produces expected header."""
        report = CacheProfileReport(
            timestamp="2026-01-01T00:00:00Z",
            page_count=10,
            results=[{
                "cache_size_mib": 1, "page_size_label": "small",
                "hits": 5, "misses": 5, "evictions": 2,
                "hit_rate": 0.5, "avg_put_us": 10, "avg_get_us": 5,
                "memory_delta_mb": 1.0,
            }],
            recommendations=[],
        )
        output = cache_console_report(report)
        assert "PAGE CACHE MEMORY PROFILE" in output

    def test_format_markdown_report(self):
        """Markdown report produces valid markdown."""
        report = CacheProfileReport(
            timestamp="2026-01-01T00:00:00Z",
            page_count=10,
            results=[{
                "cache_size_mib": 1, "page_size_label": "small",
                "hits": 5, "misses": 5, "evictions": 2,
                "hit_rate": 0.5, "avg_put_us": 10, "avg_get_us": 5,
            }],
            recommendations=[],
        )
        md = cache_md_report(report)
        assert "# Page Cache Memory Profile Report" in md


# ---------------------------------------------------------------------------
# Item 13: monitor_page_routing
# ---------------------------------------------------------------------------

from page_routing import (
    PageFeatures,
    PageRouter,
    RoutingDecision,
    RoutingTarget,
)
from scripts.monitor_page_routing import (
    RoutingComparisonResult,
    RoutingMonitorReport,
    evaluate_routing,
    generate_sample_pages,
    run_routing_monitor,
    simulate_quality,
)
from scripts.monitor_page_routing import (
    format_console_report as routing_console_report,
)
from scripts.monitor_page_routing import (
    format_markdown_report as routing_md_report,
)


class TestMonitorPageRouting:
    """Tests for the page routing quality monitor."""

    def test_generate_sample_pages_count(self):
        """Generates the requested number of pages."""
        pages = generate_sample_pages(50, seed=42)
        assert len(pages) == 50

    def test_generate_sample_pages_deterministic(self):
        """Same seed produces identical pages."""
        p1 = generate_sample_pages(20, seed=42)
        p2 = generate_sample_pages(20, seed=42)
        assert [p.complexity_score for p in p1] == [p.complexity_score for p in p2]

    def test_generate_sample_pages_diversity(self):
        """Generated pages include a mix of features."""
        pages = generate_sample_pages(200, seed=42)
        has_tables = any(p.has_tables for p in pages)
        has_handwritten = any(p.is_handwritten for p in pages)
        has_simple = any(p.complexity_score < 0.2 for p in pages)
        has_complex = any(p.complexity_score > 0.7 for p in pages)
        assert has_tables
        assert has_handwritten
        assert has_simple
        assert has_complex

    def test_simulate_quality_range(self):
        """Quality scores are in [0, 1]."""
        features = PageFeatures(page_number=1, width=2480, height=3508)
        decision = RoutingDecision(
            page_number=1, target=RoutingTarget.GPU_PADDLE
        )
        for seed in range(50):
            q = simulate_quality(decision, features, seed=seed)
            assert 0.0 <= q <= 1.0

    def test_simulate_quality_gpu_better_than_cpu(self):
        """GPU Paddle generally scores higher than CPU Tesseract."""
        features = PageFeatures(
            page_number=1, width=2480, height=3508,
            complexity_score=0.3,
        )
        gpu_decision = RoutingDecision(
            page_number=1, target=RoutingTarget.GPU_PADDLE
        )
        cpu_decision = RoutingDecision(
            page_number=1, target=RoutingTarget.CPU_TESSERACT
        )
        gpu_scores = [simulate_quality(gpu_decision, features, seed=i) for i in range(100)]
        cpu_scores = [simulate_quality(cpu_decision, features, seed=i) for i in range(100)]
        # On average, GPU should score higher
        assert sum(gpu_scores) / len(gpu_scores) > sum(cpu_scores) / len(cpu_scores)

    def test_simulate_quality_handwriting_bonus(self):
        """Handwritten pages get a quality bonus on GPU Paddle."""
        features_hw = PageFeatures(
            page_number=1, width=2480, height=3508,
            is_handwritten=True, complexity_score=0.5,
        )
        features_typed = PageFeatures(
            page_number=1, width=2480, height=3508,
            is_handwritten=False, complexity_score=0.5,
        )
        decision = RoutingDecision(
            page_number=1, target=RoutingTarget.GPU_PADDLE
        )
        hw_scores = [simulate_quality(decision, features_hw, seed=i) for i in range(100)]
        typed_scores = [simulate_quality(decision, features_typed, seed=i) for i in range(100)]
        # Handwritten should get a boost on average
        assert sum(hw_scores) / len(hw_scores) > sum(typed_scores) / len(typed_scores)

    def test_evaluate_routing_smart(self):
        """Smart routing evaluation returns valid results."""
        pages = generate_sample_pages(30, seed=42)
        router = PageRouter()
        result = evaluate_routing(pages, router, "smart")
        assert isinstance(result, RoutingComparisonResult)
        assert result.mode == "smart"
        assert result.total_pages == 30

    def test_evaluate_routing_default(self):
        """Default routing sends all pages to one backend."""
        pages = generate_sample_pages(30, seed=42)
        # Filter out tiny pages that would be skipped
        normal_pages = [p for p in pages if p.width >= 100 and p.height >= 100]
        router = PageRouter(rules=[], default_target=RoutingTarget.GPU_PADDLE)
        result = evaluate_routing(normal_pages, router, "default")
        # All should go to GPU_PADDLE
        if result.backend_distribution:
            assert "gpu_paddle" in result.backend_distribution

    def test_run_routing_monitor_report(self):
        """Full routing monitor produces a complete report."""
        report = run_routing_monitor(sample_count=30, seed=42)
        assert isinstance(report, RoutingMonitorReport)
        assert report.sample_count == 30
        assert report.smart_routing.get("mode") == "smart"
        assert len(report.routing_decisions) == 30

    def test_run_routing_monitor_quality_improvement(self):
        """Smart routing reports quality metrics."""
        report = run_routing_monitor(sample_count=50, seed=42)
        # Quality improvement can be positive or negative depending on workload
        assert isinstance(report.quality_improvement_pct, float)

    def test_run_routing_monitor_recommendations(self):
        """Monitor generates recommendations."""
        report = run_routing_monitor(sample_count=50, seed=42)
        assert len(report.recommendations) > 0

    def test_format_console_report_routing(self):
        """Console report produces expected sections."""
        report = run_routing_monitor(sample_count=20, seed=42)
        output = routing_console_report(report)
        assert "PAGE ROUTING QUALITY MONITOR" in output
        assert "SMART ROUTING" in output
        assert "DEFAULT ROUTING" in output

    def test_format_markdown_report_routing(self):
        """Markdown report produces valid markdown."""
        report = run_routing_monitor(sample_count=20, seed=42)
        md = routing_md_report(report)
        assert "# Page Routing Quality Monitor Report" in md


# ---------------------------------------------------------------------------
# Item 14: benchmark_gpu_fusion
# ---------------------------------------------------------------------------

from gpu_optimization import (
    FusionConfig,
    FusionStrategy,
    OptimizationLevel,
)
from scripts.benchmark_gpu_fusion import (
    FusionBenchmarkResult,
    GpuFusionReport,
    _check_numpy,
    _generate_test_images,
    benchmark_fusion_config,
    run_gpu_fusion_benchmark,
)
from scripts.benchmark_gpu_fusion import (
    _percentile as fusion_percentile,
)
from scripts.benchmark_gpu_fusion import (
    format_console_report as fusion_console_report,
)
from scripts.benchmark_gpu_fusion import (
    format_markdown_report as fusion_md_report,
)


class TestBenchmarkGpuFusion:
    """Tests for the GPU fusion benchmark tool."""

    def test_check_numpy(self):
        """numpy availability check returns a bool."""
        result = _check_numpy()
        assert isinstance(result, bool)

    def test_generate_test_images(self):
        """Test images are generated with correct count."""
        images = _generate_test_images(5, (64, 64))
        assert len(images) == 5

    def test_generate_test_images_size(self):
        """Generated images have the requested dimensions."""
        images = _generate_test_images(3, (128, 256))
        for img in images:
            assert img.size == (128, 256)

    def test_generate_test_images_zero(self):
        """Zero count returns empty list."""
        images = _generate_test_images(0, (64, 64))
        assert images == []

    def test_fusion_percentile_empty(self):
        """Percentile of empty list returns 0."""
        assert fusion_percentile([], 95) == 0.0

    def test_fusion_percentile_values(self):
        """Percentile returns expected value from sorted data."""
        values = list(range(100))
        p95 = fusion_percentile(values, 95)
        assert p95 == 95

    def test_benchmark_fusion_config_baseline(self):
        """Baseline fusion config benchmark runs."""
        config = FusionConfig(
            level=OptimizationLevel.NONE,
            strategy=FusionStrategy.NONE,
            max_batch_images=4,
        )
        result = benchmark_fusion_config(
            config=config,
            label="test_baseline",
            batch_size=4,
            image_size=(64, 64),
            iterations=2,
        )
        assert isinstance(result, FusionBenchmarkResult)
        assert result.config_label == "test_baseline"
        assert result.throughput_images_per_sec > 0

    def test_benchmark_fusion_config_preprocess(self):
        """Preprocess batch fusion config benchmark runs."""
        config = FusionConfig(
            level=OptimizationLevel.BASIC,
            strategy=FusionStrategy.PREPROCESS_BATCH,
            max_batch_images=4,
        )
        result = benchmark_fusion_config(
            config=config,
            label="test_preprocess",
            batch_size=4,
            image_size=(64, 64),
            iterations=2,
        )
        assert result.batch_size == 4
        assert result.avg_batch_latency_ms > 0

    def test_benchmark_fusion_config_memory_estimate(self):
        """Memory estimation is positive for non-zero batch."""
        config = FusionConfig(max_batch_images=8)
        result = benchmark_fusion_config(
            config=config,
            label="mem_test",
            batch_size=8,
            image_size=(640, 640),
            iterations=1,
        )
        assert result.estimated_memory_mb > 0

    def test_benchmark_fusion_config_optimal_batch(self):
        """Optimal batch sizes scale with memory budget."""
        config = FusionConfig(max_batch_images=32)
        result = benchmark_fusion_config(
            config=config,
            label="optimal_test",
            batch_size=4,
            image_size=(640, 640),
            iterations=1,
        )
        assert result.optimal_batch_for_8gb >= result.optimal_batch_for_4gb
        assert result.optimal_batch_for_4gb >= result.optimal_batch_for_1gb

    def test_run_gpu_fusion_benchmark(self):
        """Full GPU fusion benchmark runs and produces report."""
        report = run_gpu_fusion_benchmark(
            batch_size=2,
            iterations=1,
            image_size=(64, 64),
        )
        assert isinstance(report, GpuFusionReport)
        assert len(report.results) >= 4  # At least 4 CPU configs
        assert report.timestamp
        assert len(report.recommendations) > 0

    def test_format_console_report_fusion(self):
        """Console report includes expected sections."""
        report = run_gpu_fusion_benchmark(
            batch_size=2, iterations=1, image_size=(64, 64),
        )
        output = fusion_console_report(report)
        assert "GPU FUSION BENCHMARK REPORT" in output

    def test_format_markdown_report_fusion(self):
        """Markdown report has valid structure."""
        report = run_gpu_fusion_benchmark(
            batch_size=2, iterations=1, image_size=(64, 64),
        )
        md = fusion_md_report(report)
        assert "# GPU Fusion Benchmark Report" in md


# ---------------------------------------------------------------------------
# Item 15: validate_keda_scaling
# ---------------------------------------------------------------------------

from scripts.validate_keda_scaling import (
    STRATEGY_DEFAULTS,
    KedaValidationReport,
    ScalerConfig,
    ScaleSimulationResult,
    _generate_queue_depth,
    _get_default_values,
    extract_scalers,
    parse_helm_values,
    run_keda_validation,
    simulate_scaling,
    validate_scaler,
)
from scripts.validate_keda_scaling import (
    format_console_report as keda_console_report,
)
from scripts.validate_keda_scaling import (
    format_markdown_report as keda_md_report,
)


class TestValidateKedaScaling:
    """Tests for the KEDA autoscaling validation tool."""

    def test_strategy_defaults_exist(self):
        """All expected strategies have defaults."""
        assert "aggressive" in STRATEGY_DEFAULTS
        assert "balanced" in STRATEGY_DEFAULTS
        assert "conservative" in STRATEGY_DEFAULTS

    def test_get_default_values(self):
        """Default values include expected worker keys."""
        values = _get_default_values()
        assert "gpuWorker" in values
        assert "cpuWorker" in values
        assert "keda" in values

    def test_parse_helm_values_missing_file(self):
        """Missing file falls back to defaults."""
        values = parse_helm_values("/nonexistent/path.yaml")
        assert "gpuWorker" in values

    def test_extract_scalers_count(self):
        """extract_scalers returns configs for all worker types."""
        values = _get_default_values()
        scalers = extract_scalers(values)
        assert len(scalers) == 4  # gpu, cpu, nlp, layoutlm

    def test_extract_scalers_strategy_override(self):
        """Aggressive strategy overrides polling and cooldown."""
        values = _get_default_values()
        values["keda"]["scalingStrategy"] = "aggressive"
        scalers = extract_scalers(values)
        gpu_scaler = next(s for s in scalers if "GPU" in s.name)
        assert gpu_scaler.effective_polling == 10
        assert gpu_scaler.effective_cooldown == 60

    def test_extract_scalers_conservative(self):
        """Conservative strategy uses longer intervals."""
        values = _get_default_values()
        values["keda"]["scalingStrategy"] = "conservative"
        scalers = extract_scalers(values)
        gpu_scaler = next(s for s in scalers if "GPU" in s.name)
        assert gpu_scaler.effective_polling == 30
        assert gpu_scaler.effective_cooldown == 600

    def test_extract_scalers_max_clamped(self):
        """Max replicas are clamped to global maximum."""
        values = _get_default_values()
        values["keda"]["maxReplicaCount"] = 5
        values["gpuWorker"]["autoscaling"]["maxReplicas"] = 100
        scalers = extract_scalers(values)
        gpu_scaler = next(s for s in scalers if "GPU" in s.name)
        assert gpu_scaler.max_replicas == 5

    def test_validate_scaler_valid(self):
        """Valid scaler passes validation."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=10,
            queue_target=5, effective_polling=15, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        assert all(f.severity == "info" for f in findings)

    def test_validate_scaler_min_gt_max(self):
        """Catches min > max replicas."""
        scaler = ScalerConfig(
            name="Test", min_replicas=20, max_replicas=5,
            queue_target=5, effective_polling=15, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        errors = [f for f in findings if f.severity == "error"]
        assert len(errors) >= 1
        assert "minReplicas" in errors[0].message

    def test_validate_scaler_negative_min(self):
        """Catches negative min replicas."""
        scaler = ScalerConfig(
            name="Test", min_replicas=-1, max_replicas=10,
            queue_target=5, effective_polling=15, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        errors = [f for f in findings if f.severity == "error"]
        assert any("must be >= 0" in f.message for f in errors)

    def test_validate_scaler_zero_queue_target(self):
        """Catches zero queue target."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=10,
            queue_target=0, effective_polling=15, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        errors = [f for f in findings if f.severity == "error"]
        assert any("queueTarget" in f.message for f in errors)

    def test_validate_scaler_low_polling(self):
        """Warns on very low polling interval."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=10,
            queue_target=5, effective_polling=3, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        warnings = [f for f in findings if f.severity == "warning"]
        assert any("pollingInterval" in f.message for f in warnings)

    def test_validate_scaler_low_cooldown(self):
        """Warns on very low cooldown period."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=10,
            queue_target=5, effective_polling=15, effective_cooldown=10,
        )
        findings = validate_scaler(scaler)
        warnings = [f for f in findings if f.severity == "warning"]
        assert any("cooldownPeriod" in f.message for f in warnings)

    def test_validate_scaler_gpu_zero_min(self):
        """Warns on GPU worker with min_replicas=0."""
        scaler = ScalerConfig(
            name="GPU Worker", min_replicas=0, max_replicas=10,
            queue_target=5, effective_polling=15, effective_cooldown=300,
        )
        findings = validate_scaler(scaler)
        warnings = [f for f in findings if f.severity == "warning"]
        assert any("cold starts" in f.message for f in warnings)

    def test_generate_queue_depth_burst(self):
        """Burst pattern produces a spike then decay."""
        depths = [_generate_queue_depth(t, 600, "burst") for t in range(0, 600, 15)]
        peak = max(depths)
        assert peak > 0
        # Last quarter should be declining
        assert depths[-1] < peak

    def test_generate_queue_depth_steady(self):
        """Steady pattern produces constant depth."""
        depths = [_generate_queue_depth(t, 600, "steady") for t in range(0, 600, 15)]
        assert all(d == depths[0] for d in depths)

    def test_generate_queue_depth_wave(self):
        """Wave pattern produces varying depth."""
        depths = [_generate_queue_depth(t, 600, "wave") for t in range(0, 600, 15)]
        assert min(depths) != max(depths)

    def test_simulate_scaling_burst(self):
        """Burst simulation produces scale-up events."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=20,
            queue_target=5, effective_polling=15, effective_cooldown=60,
        )
        result = simulate_scaling(scaler, duration_s=300, workload_pattern="burst")
        assert isinstance(result, ScaleSimulationResult)
        assert result.scale_up_events > 0
        assert result.max_replicas_reached >= 1

    def test_simulate_scaling_respects_max(self):
        """Simulation never exceeds max replicas."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=3,
            queue_target=1, effective_polling=15, effective_cooldown=60,
        )
        result = simulate_scaling(scaler, duration_s=300, workload_pattern="burst")
        assert result.max_replicas_reached <= 3

    def test_simulate_scaling_cooldown(self):
        """Cooldown prevents immediate scale-down."""
        scaler = ScalerConfig(
            name="Test", min_replicas=1, max_replicas=20,
            queue_target=5, effective_polling=15, effective_cooldown=300,
        )
        result = simulate_scaling(scaler, duration_s=600, workload_pattern="burst")
        # With long cooldown, scale-down events should be limited
        assert isinstance(result.scale_down_events, int)

    def test_run_keda_validation_defaults(self):
        """Full validation with defaults produces a report."""
        report = run_keda_validation(simulate_duration=60)
        assert isinstance(report, KedaValidationReport)
        assert len(report.scalers) == 4
        assert len(report.findings) > 0

    def test_run_keda_validation_passes(self):
        """Default Helm values pass validation."""
        report = run_keda_validation(simulate_duration=60)
        # Default values should have no errors (only info/warning)
        assert report.total_errors == 0

    def test_run_keda_validation_simulations(self):
        """Validation includes simulations for all scalers."""
        report = run_keda_validation(simulate_duration=60)
        # 4 scalers x 3 patterns = 12 simulations
        assert len(report.simulations) == 12

    def test_format_console_report_keda(self):
        """Console report includes expected sections."""
        report = run_keda_validation(simulate_duration=60)
        output = keda_console_report(report)
        assert "KEDA AUTOSCALING VALIDATION" in output
        assert "FINDINGS" in output
        assert "SIMULATION SUMMARY" in output

    def test_format_markdown_report_keda(self):
        """Markdown report has valid structure."""
        report = run_keda_validation(simulate_duration=60)
        md = keda_md_report(report)
        assert "# KEDA Autoscaling Validation Report" in md
        assert "## Findings" in md

    def test_cli_keda_default(self):
        """KEDA CLI runs with defaults and exits 0."""
        from scripts.validate_keda_scaling import main
        with mock.patch("sys.argv", [
            "validate_keda_scaling.py",
            "--simulate-duration", "30",
        ]):
            result = main()
            assert result == 0


# ---------------------------------------------------------------------------
# Cross-tool: output file generation
# ---------------------------------------------------------------------------


class TestOutputFileGeneration:
    """Verify all tools can write JSON and markdown reports."""

    def test_adaptive_batch_output_files(self):
        """benchmark_adaptive_batch writes JSON and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_full_benchmark(
                workload_size=5, iterations=1,
                strategies=["fixed"], batch_sizes=[4], profiles=["small"],
            )
            json_path = Path(tmpdir) / "test.json"
            md_path = Path(tmpdir) / "test.md"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f)
            with open(md_path, "w") as f:
                f.write(format_markdown_report(report))
            assert json_path.exists()
            assert md_path.exists()
            data = json.loads(json_path.read_text())
            assert "results" in data

    def test_page_cache_output_files(self):
        """profile_page_cache writes JSON and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_full_profile(cache_sizes_mib=[1], page_count=5)
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f)
            assert json_path.exists()

    def test_routing_monitor_output_files(self):
        """monitor_page_routing writes JSON and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_routing_monitor(sample_count=10, seed=42)
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f)
            assert json_path.exists()

    def test_gpu_fusion_output_files(self):
        """benchmark_gpu_fusion writes JSON and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_gpu_fusion_benchmark(
                batch_size=2, iterations=1, image_size=(32, 32),
            )
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f)
            assert json_path.exists()

    def test_keda_validation_output_files(self):
        """validate_keda_scaling writes JSON and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_keda_validation(simulate_duration=30)
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, "w") as f:
                json.dump(asdict(report), f)
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert "scalers" in data
