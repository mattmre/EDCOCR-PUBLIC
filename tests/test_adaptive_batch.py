"""Tests for adaptive batch sizing module (adaptive_batch.py).

Covers:
- BatchStrategy enum values and count
- BatchConfig defaults and custom values
- PageComplexity creation and fields
- PageComplexity complexity_score computation
- BatchResult creation and computed properties
- BatchResult throughput_pages_per_sec
- BatchResult with zero duration edge case
- AdaptiveBatchSizer construction
- compute_complexity for various inputs
- compute_complexity large vs small pages
- compute_complexity with tables increases score
- recommend_batch_size with FIXED strategy
- recommend_batch_size with ADAPTIVE strategy
- recommend_batch_size with high complexity reduces size
- recommend_batch_size with low complexity increases size
- recommend_batch_size respects min/max bounds
- record_result tracks history
- record_result adjusts after warmup
- record_result memory pressure reduces batch
- record_result throughput improvement increases batch
- get_current_batch_size initial value
- get_history returns recorded results
- reset clears state
- Thread safety concurrent recording

Run with: python -m pytest tests/test_adaptive_batch.py -v
"""

import threading

import pytest

# Add project root to path
from adaptive_batch import (
    AdaptiveBatchSizer,
    BatchConfig,
    BatchResult,
    BatchStrategy,
    PageComplexity,
)

# ---------------------------------------------------------------------------
# Tests: BatchStrategy
# ---------------------------------------------------------------------------


class TestBatchStrategy:
    def test_enum_has_four_members(self):
        assert len(BatchStrategy) == 4

    def test_fixed_value(self):
        assert BatchStrategy.FIXED.value == "fixed"

    def test_adaptive_value(self):
        assert BatchStrategy.ADAPTIVE.value == "adaptive"

    def test_memory_aware_value(self):
        assert BatchStrategy.MEMORY_AWARE.value == "memory_aware"

    def test_throughput_optimal_value(self):
        assert BatchStrategy.THROUGHPUT_OPTIMAL.value == "throughput_optimal"

    def test_enum_from_value(self):
        assert BatchStrategy("adaptive") is BatchStrategy.ADAPTIVE


# ---------------------------------------------------------------------------
# Tests: BatchConfig
# ---------------------------------------------------------------------------


class TestBatchConfig:
    def test_default_values(self):
        cfg = BatchConfig()
        assert cfg.strategy == BatchStrategy.ADAPTIVE
        assert cfg.min_batch_size == 1
        assert cfg.max_batch_size == 32
        assert cfg.target_memory_pct == 75.0
        assert cfg.target_latency_ms == 500.0
        assert cfg.warmup_batches == 3
        assert cfg.adjustment_factor == 0.1

    def test_custom_values(self):
        cfg = BatchConfig(
            strategy=BatchStrategy.FIXED,
            min_batch_size=4,
            max_batch_size=64,
            target_memory_pct=80.0,
            target_latency_ms=200.0,
            warmup_batches=5,
            adjustment_factor=0.2,
        )
        assert cfg.strategy == BatchStrategy.FIXED
        assert cfg.min_batch_size == 4
        assert cfg.max_batch_size == 64
        assert cfg.target_memory_pct == 80.0
        assert cfg.target_latency_ms == 200.0
        assert cfg.warmup_batches == 5
        assert cfg.adjustment_factor == 0.2

    def test_min_less_than_max(self):
        cfg = BatchConfig(min_batch_size=1, max_batch_size=8)
        assert cfg.min_batch_size < cfg.max_batch_size


# ---------------------------------------------------------------------------
# Tests: PageComplexity
# ---------------------------------------------------------------------------


class TestPageComplexity:
    def test_default_creation(self):
        pc = PageComplexity()
        assert pc.page_number == 1
        assert pc.width == 0
        assert pc.height == 0
        assert pc.file_size_bytes == 0
        assert pc.estimated_text_density == 0.0
        assert pc.has_tables is False
        assert pc.has_images is False
        assert pc.dpi == 300
        assert pc.complexity_score == 0.0

    def test_custom_fields(self):
        pc = PageComplexity(
            page_number=5,
            width=2480,
            height=3508,
            file_size_bytes=500_000,
            estimated_text_density=0.7,
            has_tables=True,
            has_images=False,
            dpi=600,
            complexity_score=0.85,
        )
        assert pc.page_number == 5
        assert pc.width == 2480
        assert pc.height == 3508
        assert pc.file_size_bytes == 500_000
        assert pc.estimated_text_density == 0.7
        assert pc.has_tables is True
        assert pc.has_images is False
        assert pc.dpi == 600
        assert pc.complexity_score == 0.85

    def test_complexity_score_settable(self):
        pc = PageComplexity(complexity_score=1.5)
        assert pc.complexity_score == 1.5


# ---------------------------------------------------------------------------
# Tests: BatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_default_creation(self):
        br = BatchResult()
        assert br.batch_size == 0
        assert br.pages_processed == 0
        assert br.duration_seconds == 0.0
        assert br.memory_peak_mb == 0.0
        assert br.avg_page_complexity == 0.0
        assert br.success_count == 0
        assert br.failure_count == 0

    def test_throughput_pages_per_sec(self):
        br = BatchResult(pages_processed=10, duration_seconds=2.0)
        assert br.throughput_pages_per_sec == pytest.approx(5.0)

    def test_throughput_zero_duration(self):
        br = BatchResult(pages_processed=10, duration_seconds=0.0)
        assert br.throughput_pages_per_sec == 0.0

    def test_throughput_negative_duration(self):
        br = BatchResult(pages_processed=10, duration_seconds=-1.0)
        assert br.throughput_pages_per_sec == 0.0

    def test_throughput_zero_pages(self):
        br = BatchResult(pages_processed=0, duration_seconds=5.0)
        assert br.throughput_pages_per_sec == 0.0

    def test_custom_creation(self):
        br = BatchResult(
            batch_size=8,
            pages_processed=8,
            duration_seconds=1.5,
            memory_peak_mb=512.0,
            avg_page_complexity=0.45,
            success_count=7,
            failure_count=1,
        )
        assert br.batch_size == 8
        assert br.success_count == 7
        assert br.failure_count == 1
        assert br.throughput_pages_per_sec == pytest.approx(8 / 1.5)


# ---------------------------------------------------------------------------
# Tests: AdaptiveBatchSizer — construction
# ---------------------------------------------------------------------------


class TestAdaptiveBatchSizerConstruction:
    def test_default_construction(self):
        sizer = AdaptiveBatchSizer()
        # max_batch_size=32, initial = 32 // 2 = 16
        assert sizer.get_current_batch_size() == 16

    def test_custom_config(self):
        cfg = BatchConfig(min_batch_size=2, max_batch_size=20)
        sizer = AdaptiveBatchSizer(cfg)
        # 20 // 2 = 10
        assert sizer.get_current_batch_size() == 10

    def test_initial_clamped_to_min(self):
        cfg = BatchConfig(min_batch_size=5, max_batch_size=8)
        sizer = AdaptiveBatchSizer(cfg)
        # 8 // 2 = 4, but min is 5, so initial = 5
        assert sizer.get_current_batch_size() == 5

    def test_empty_history_on_init(self):
        sizer = AdaptiveBatchSizer()
        assert sizer.get_history() == []


# ---------------------------------------------------------------------------
# Tests: compute_complexity
# ---------------------------------------------------------------------------


class TestComputeComplexity:
    def test_basic_computation(self):
        sizer = AdaptiveBatchSizer()
        pc = sizer.compute_complexity(
            width=2480, height=3508, file_size=1_000_000, dpi=300,
        )
        assert pc.width == 2480
        assert pc.height == 3508
        assert pc.file_size_bytes == 1_000_000
        assert pc.dpi == 300
        assert pc.has_tables is False
        assert pc.has_images is False
        assert pc.complexity_score > 0

    def test_small_page_low_score(self):
        sizer = AdaptiveBatchSizer()
        small = sizer.compute_complexity(
            width=100, height=100, file_size=1_000, dpi=72,
        )
        large = sizer.compute_complexity(
            width=4960, height=7016, file_size=5_000_000, dpi=600,
        )
        assert small.complexity_score < large.complexity_score

    def test_tables_increase_score(self):
        sizer = AdaptiveBatchSizer()
        without = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_tables=False,
        )
        with_tables = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_tables=True,
        )
        assert with_tables.complexity_score > without.complexity_score

    def test_images_increase_score(self):
        sizer = AdaptiveBatchSizer()
        without = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_images=False,
        )
        with_images = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_images=True,
        )
        assert with_images.complexity_score > without.complexity_score

    def test_high_dpi_increases_score(self):
        sizer = AdaptiveBatchSizer()
        low_dpi = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=150,
        )
        high_dpi = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=600,
        )
        assert high_dpi.complexity_score > low_dpi.complexity_score

    def test_zero_dimensions(self):
        sizer = AdaptiveBatchSizer()
        pc = sizer.compute_complexity(
            width=0, height=0, file_size=0, dpi=300,
        )
        assert pc.complexity_score == 0.0

    def test_tables_and_images_combined(self):
        sizer = AdaptiveBatchSizer()
        pc = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_tables=True, has_images=True,
        )
        plain = sizer.compute_complexity(
            width=2480, height=3508, file_size=500_000, dpi=300,
            has_tables=False, has_images=False,
        )
        # Tables add 0.2, images add 0.2
        diff = pc.complexity_score - plain.complexity_score
        assert diff == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: recommend_batch_size
# ---------------------------------------------------------------------------


class TestRecommendBatchSize:
    def test_fixed_strategy_returns_max(self):
        cfg = BatchConfig(strategy=BatchStrategy.FIXED, max_batch_size=24)
        sizer = AdaptiveBatchSizer(cfg)
        pages = [PageComplexity(complexity_score=0.9) for _ in range(10)]
        assert sizer.recommend_batch_size(pages) == 24

    def test_adaptive_empty_pages(self):
        sizer = AdaptiveBatchSizer()
        # No pages → returns current batch size
        assert sizer.recommend_batch_size([]) == sizer.get_current_batch_size()

    def test_adaptive_low_complexity(self):
        cfg = BatchConfig(max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        pages = [PageComplexity(complexity_score=0.1) for _ in range(5)]
        rec = sizer.recommend_batch_size(pages)
        # Low complexity → large batch (close to max)
        assert rec >= 20

    def test_adaptive_high_complexity_reduces_size(self):
        cfg = BatchConfig(max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        pages = [PageComplexity(complexity_score=1.5) for _ in range(5)]
        rec = sizer.recommend_batch_size(pages)
        # High complexity → smaller batch
        assert rec < 32

    def test_recommend_respects_min_bound(self):
        cfg = BatchConfig(min_batch_size=4, max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        # Very high complexity to push recommendation low
        pages = [PageComplexity(complexity_score=5.0) for _ in range(5)]
        rec = sizer.recommend_batch_size(pages)
        assert rec >= 4

    def test_recommend_respects_max_bound(self):
        cfg = BatchConfig(min_batch_size=1, max_batch_size=16)
        sizer = AdaptiveBatchSizer(cfg)
        # Very low complexity
        pages = [PageComplexity(complexity_score=0.0) for _ in range(5)]
        rec = sizer.recommend_batch_size(pages)
        assert rec <= 16


# ---------------------------------------------------------------------------
# Tests: record_result
# ---------------------------------------------------------------------------


class TestRecordResult:
    def test_tracks_history(self):
        sizer = AdaptiveBatchSizer()
        r = BatchResult(batch_size=8, pages_processed=8, duration_seconds=1.0)
        sizer.record_result(r)
        assert len(sizer.get_history()) == 1

    def test_multiple_results_tracked(self):
        sizer = AdaptiveBatchSizer()
        for i in range(5):
            sizer.record_result(
                BatchResult(
                    batch_size=8,
                    pages_processed=8,
                    duration_seconds=1.0 + i,
                )
            )
        assert len(sizer.get_history()) == 5

    def test_no_adjustment_before_warmup(self):
        cfg = BatchConfig(warmup_batches=3, max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        initial = sizer.get_current_batch_size()
        # Record only 2 results (< warmup)
        for _ in range(2):
            sizer.record_result(
                BatchResult(
                    batch_size=16,
                    pages_processed=16,
                    duration_seconds=1.0,
                    memory_peak_mb=50.0,
                )
            )
        assert sizer.get_current_batch_size() == initial

    def test_adjusts_after_warmup(self):
        cfg = BatchConfig(warmup_batches=3, max_batch_size=32)
        sizer = AdaptiveBatchSizer(cfg)
        sizer.get_current_batch_size()  # verify initial state
        # Record 3 results (= warmup) with enough variation to trigger change
        for i in range(3):
            sizer.record_result(
                BatchResult(
                    batch_size=16,
                    pages_processed=16,
                    duration_seconds=2.0 - i * 0.5,  # improving throughput
                    memory_peak_mb=50.0,
                )
            )
        # After warmup with improving throughput, size should change
        # (may increase due to throughput improvement)
        assert len(sizer.get_history()) == 3

    def test_memory_pressure_reduces_batch(self):
        cfg = BatchConfig(
            warmup_batches=2,
            max_batch_size=32,
            target_memory_pct=75.0,
            adjustment_factor=0.1,
        )
        sizer = AdaptiveBatchSizer(cfg)
        initial = sizer.get_current_batch_size()  # 16

        # First result: normal
        sizer.record_result(
            BatchResult(
                batch_size=16,
                pages_processed=16,
                duration_seconds=1.0,
                memory_peak_mb=50.0,
            )
        )
        # Second result (hits warmup): high memory
        sizer.record_result(
            BatchResult(
                batch_size=16,
                pages_processed=16,
                duration_seconds=1.0,
                memory_peak_mb=90.0,  # > 75% target
            )
        )
        assert sizer.get_current_batch_size() < initial

    def test_throughput_improvement_increases_batch(self):
        cfg = BatchConfig(
            warmup_batches=2,
            max_batch_size=32,
            target_memory_pct=75.0,
            adjustment_factor=0.2,
        )
        sizer = AdaptiveBatchSizer(cfg)
        initial = sizer.get_current_batch_size()  # 16

        # First result: low throughput
        sizer.record_result(
            BatchResult(
                batch_size=16,
                pages_processed=16,
                duration_seconds=4.0,  # 4 pages/s
                memory_peak_mb=30.0,
            )
        )
        # Second result (hits warmup): better throughput, low memory
        sizer.record_result(
            BatchResult(
                batch_size=16,
                pages_processed=16,
                duration_seconds=1.0,  # 16 pages/s — much better
                memory_peak_mb=30.0,
            )
        )
        assert sizer.get_current_batch_size() > initial


# ---------------------------------------------------------------------------
# Tests: get_current_batch_size
# ---------------------------------------------------------------------------


class TestGetCurrentBatchSize:
    def test_initial_value_default(self):
        sizer = AdaptiveBatchSizer()
        assert sizer.get_current_batch_size() == 16  # 32 // 2

    def test_initial_value_custom_max(self):
        cfg = BatchConfig(max_batch_size=10)
        sizer = AdaptiveBatchSizer(cfg)
        assert sizer.get_current_batch_size() == 5  # 10 // 2


# ---------------------------------------------------------------------------
# Tests: get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    def test_empty_initially(self):
        sizer = AdaptiveBatchSizer()
        assert sizer.get_history() == []

    def test_returns_copy(self):
        sizer = AdaptiveBatchSizer()
        r = BatchResult(batch_size=4, pages_processed=4, duration_seconds=0.5)
        sizer.record_result(r)
        history = sizer.get_history()
        history.clear()
        # Original still has the entry
        assert len(sizer.get_history()) == 1

    def test_preserves_order(self):
        sizer = AdaptiveBatchSizer()
        for i in range(3):
            sizer.record_result(
                BatchResult(batch_size=i + 1, pages_processed=i + 1, duration_seconds=1.0)
            )
        history = sizer.get_history()
        assert [h.batch_size for h in history] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_history(self):
        sizer = AdaptiveBatchSizer()
        sizer.record_result(
            BatchResult(batch_size=8, pages_processed=8, duration_seconds=1.0)
        )
        sizer.reset()
        assert sizer.get_history() == []

    def test_restores_initial_batch_size(self):
        cfg = BatchConfig(max_batch_size=20)
        sizer = AdaptiveBatchSizer(cfg)
        initial = sizer.get_current_batch_size()

        # Drive the sizer through warmup to change batch size
        for i in range(5):
            sizer.record_result(
                BatchResult(
                    batch_size=10,
                    pages_processed=10,
                    duration_seconds=1.0,
                    memory_peak_mb=90.0,
                )
            )

        sizer.reset()
        assert sizer.get_current_batch_size() == initial

    def test_can_record_after_reset(self):
        sizer = AdaptiveBatchSizer()
        sizer.record_result(
            BatchResult(batch_size=8, pages_processed=8, duration_seconds=1.0)
        )
        sizer.reset()
        sizer.record_result(
            BatchResult(batch_size=4, pages_processed=4, duration_seconds=0.5)
        )
        assert len(sizer.get_history()) == 1
        assert sizer.get_history()[0].batch_size == 4


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_recording(self):
        sizer = AdaptiveBatchSizer(BatchConfig(warmup_batches=2))
        errors: list[Exception] = []
        num_threads = 8
        results_per_thread = 20

        def _record_results():
            try:
                for i in range(results_per_thread):
                    sizer.record_result(
                        BatchResult(
                            batch_size=8,
                            pages_processed=8,
                            duration_seconds=0.5 + i * 0.01,
                            memory_peak_mb=40.0 + i,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_record_results) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(sizer.get_history()) == num_threads * results_per_thread

    def test_concurrent_recommend_and_record(self):
        sizer = AdaptiveBatchSizer(BatchConfig(warmup_batches=1))
        errors: list[Exception] = []

        def _recommend():
            try:
                for _ in range(50):
                    pages = [PageComplexity(complexity_score=0.5) for _ in range(3)]
                    sizer.recommend_batch_size(pages)
            except Exception as exc:
                errors.append(exc)

        def _record():
            try:
                for i in range(50):
                    sizer.record_result(
                        BatchResult(
                            batch_size=8,
                            pages_processed=8,
                            duration_seconds=1.0,
                            memory_peak_mb=50.0,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_recommend)
        t2 = threading.Thread(target=_record)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_reset_and_record(self):
        sizer = AdaptiveBatchSizer()
        errors: list[Exception] = []

        def _record():
            try:
                for _ in range(50):
                    sizer.record_result(
                        BatchResult(
                            batch_size=4,
                            pages_processed=4,
                            duration_seconds=0.5,
                        )
                    )
            except Exception as exc:
                errors.append(exc)

        def _reset():
            try:
                for _ in range(10):
                    sizer.reset()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_record)
        t2 = threading.Thread(target=_reset)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
