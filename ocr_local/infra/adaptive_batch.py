"""Adaptive batch sizing for the OCR pipeline.

Dynamically adjusts batch sizes based on document complexity, available
memory, and processing throughput.  Supports four strategies: FIXED
(static batch size), ADAPTIVE (complexity-aware), MEMORY_AWARE (memory-
pressure feedback), and THROUGHPUT_OPTIMAL (maximise pages/second).

The :class:`AdaptiveBatchSizer` maintains a rolling history of
:class:`BatchResult` records and automatically tunes the current batch
size after a configurable warmup period.

Environment Variables:
    ADAPTIVE_BATCH_STRATEGY (str):
        Batch strategy: fixed, adaptive, memory_aware, or
        throughput_optimal.  Default: ``adaptive``.
    ADAPTIVE_BATCH_MAX (int):
        Maximum batch size.  Default: ``32``.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BatchStrategy(Enum):
    """Available batch sizing strategies."""

    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    MEMORY_AWARE = "memory_aware"
    THROUGHPUT_OPTIMAL = "throughput_optimal"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """Configuration for the adaptive batch sizer.

    Attributes:
        strategy: The batch sizing strategy to use.
        min_batch_size: Minimum number of pages in a batch.
        max_batch_size: Maximum number of pages in a batch.
        target_memory_pct: Target memory utilisation percentage (0–100).
        target_latency_ms: Target per-batch latency in milliseconds.
        warmup_batches: Number of batches processed before adaptation begins.
        adjustment_factor: Fractional step size for batch size adjustments.
    """

    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_memory_pct: float = 75.0
    target_latency_ms: float = 500.0
    warmup_batches: int = 3
    adjustment_factor: float = 0.1


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# Weights used in complexity scoring
_AREA_WEIGHT = 0.3
_FILE_SIZE_WEIGHT = 0.3
_TABLE_WEIGHT = 0.2
_IMAGE_WEIGHT = 0.2

# Reference values for normalisation
_REF_AREA = 8_500_000  # ~A4 at 300 DPI (2480 × 3508 ≈ 8.7M)
_REF_FILE_SIZE = 2_000_000  # 2 MB


@dataclass
class PageComplexity:
    """Complexity profile of a single page.

    Attributes:
        page_number: 1-based page index within the document.
        width: Page width in pixels.
        height: Page height in pixels.
        file_size_bytes: Size of the source file in bytes.
        estimated_text_density: Approximate text coverage ratio (0–1).
        has_tables: Whether the page contains table structures.
        has_images: Whether the page contains embedded images.
        dpi: Effective scan resolution.
        complexity_score: Overall complexity score.  If not explicitly set,
            this should be populated by
            :meth:`AdaptiveBatchSizer.compute_complexity`.
    """

    page_number: int = 1
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    estimated_text_density: float = 0.0
    has_tables: bool = False
    has_images: bool = False
    dpi: int = 300
    complexity_score: float = 0.0


@dataclass
class BatchResult:
    """Outcome metrics for a single processed batch.

    Attributes:
        batch_size: Number of pages submitted in this batch.
        pages_processed: Number of pages that completed processing.
        duration_seconds: Wall-clock time for the batch.
        memory_peak_mb: Peak memory usage during the batch in megabytes.
        avg_page_complexity: Mean complexity score across batch pages.
        success_count: Number of pages that succeeded.
        failure_count: Number of pages that failed.
    """

    batch_size: int = 0
    pages_processed: int = 0
    duration_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    avg_page_complexity: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    @property
    def throughput_pages_per_sec(self) -> float:
        """Compute throughput as pages processed per second.

        Returns ``0.0`` when *duration_seconds* is zero or negative to avoid
        division-by-zero.
        """
        if self.duration_seconds <= 0:
            return 0.0
        return self.pages_processed / self.duration_seconds


# ---------------------------------------------------------------------------
# Adaptive batch sizer
# ---------------------------------------------------------------------------


class AdaptiveBatchSizer:
    """Thread-safe adaptive batch sizing engine.

    The sizer starts at ``max_batch_size // 2`` (or ``min_batch_size`` when
    the computed midpoint would be below it) and refines the batch size
    after every :pyattr:`BatchConfig.warmup_batches` results have been
    recorded.

    Adaptation rules (applied on each :meth:`record_result` call after
    warmup):

    * **Memory pressure** — if the most recent ``memory_peak_mb`` exceeds
      ``target_memory_pct`` percent of an estimated system budget, the
      batch size is decreased by ``adjustment_factor``.
    * **Throughput trend** — if the latest throughput exceeds the average
      throughput of all previous results, the batch size is increased by
      ``adjustment_factor``.
    * **High complexity** — when ``avg_page_complexity`` exceeds 0.6 the
      increase step is halved, biasing toward smaller batches.
    * The result is always clamped to ``[min_batch_size, max_batch_size]``.
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        self._config = config or BatchConfig()
        self._lock = threading.Lock()
        self._history: list[BatchResult] = []

        # Initial batch size: midpoint of range
        initial = max(
            self._config.min_batch_size,
            self._config.max_batch_size // 2,
        )
        self._current_batch_size: int = min(
            initial, self._config.max_batch_size
        )

    # -- public API ---------------------------------------------------------

    def compute_complexity(
        self,
        width: int,
        height: int,
        file_size: int,
        dpi: int = 300,
        has_tables: bool = False,
        has_images: bool = False,
    ) -> PageComplexity:
        """Compute a :class:`PageComplexity` from raw page attributes.

        The ``complexity_score`` is a weighted blend of normalised area,
        normalised file size, table presence, and image presence.
        """
        area = width * height
        normalised_area = min(area / _REF_AREA, 2.0)
        normalised_size = min(file_size / _REF_FILE_SIZE, 2.0)

        # DPI factor — scale area proportionally for very high-res scans
        dpi_factor = dpi / 300.0
        normalised_area *= min(dpi_factor, 2.0)

        table_flag = 1.0 if has_tables else 0.0
        image_flag = 1.0 if has_images else 0.0

        score = (
            _AREA_WEIGHT * normalised_area
            + _FILE_SIZE_WEIGHT * normalised_size
            + _TABLE_WEIGHT * table_flag
            + _IMAGE_WEIGHT * image_flag
        )

        return PageComplexity(
            width=width,
            height=height,
            file_size_bytes=file_size,
            dpi=dpi,
            has_tables=has_tables,
            has_images=has_images,
            complexity_score=round(score, 4),
        )

    def recommend_batch_size(self, pages: list[PageComplexity]) -> int:
        """Recommend a batch size for the given list of page profiles.

        For the ``FIXED`` strategy the configured ``max_batch_size`` is
        always returned.  For adaptive strategies the recommendation is
        derived from page complexity and internal state.
        """
        with self._lock:
            if self._config.strategy == BatchStrategy.FIXED:
                return self._config.max_batch_size

            if not pages:
                return self._current_batch_size

            avg_complexity = sum(p.complexity_score for p in pages) / len(pages)

            # Higher complexity → proportionally smaller batches
            if avg_complexity > 0:
                ratio = max(0.2, 1.0 - avg_complexity * 0.5)
            else:
                ratio = 1.0

            recommended = int(self._config.max_batch_size * ratio)
            return self._clamp(recommended)

    def record_result(self, result: BatchResult) -> None:
        """Record a batch result and adapt the batch size if appropriate."""
        with self._lock:
            self._history.append(result)

            if len(self._history) < self._config.warmup_batches:
                return

            self._adapt(result)

    def get_current_batch_size(self) -> int:
        """Return the current internal batch size."""
        with self._lock:
            return self._current_batch_size

    def get_history(self) -> list[BatchResult]:
        """Return a copy of all recorded batch results."""
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        """Reset all internal state to initial values."""
        with self._lock:
            self._history.clear()
            initial = max(
                self._config.min_batch_size,
                self._config.max_batch_size // 2,
            )
            self._current_batch_size = min(
                initial, self._config.max_batch_size
            )

    # -- internals ----------------------------------------------------------

    def _adapt(self, latest: BatchResult) -> None:
        """Adjust ``_current_batch_size`` based on the latest result.

        Must be called while ``_lock`` is held.
        """
        cfg = self._config
        current = self._current_batch_size
        step = max(1, int(current * cfg.adjustment_factor))

        # Memory pressure: reduce batch size
        if latest.memory_peak_mb > 0:
            # Use memory_peak_mb as percentage proxy (0-100 scale)
            if latest.memory_peak_mb > cfg.target_memory_pct:
                current -= step
                logger.debug(
                    "Memory pressure %.1f%% > target %.1f%% — reducing batch by %d",
                    latest.memory_peak_mb,
                    cfg.target_memory_pct,
                    step,
                )

        # Throughput trend: increase if improving
        prior_results = self._history[:-1]
        if prior_results:
            prior_throughputs = [
                r.throughput_pages_per_sec
                for r in prior_results
                if r.throughput_pages_per_sec > 0
            ]
            if prior_throughputs:
                avg_prior = sum(prior_throughputs) / len(prior_throughputs)
                if latest.throughput_pages_per_sec > avg_prior:
                    inc = step
                    # Dampen increase for high-complexity batches
                    if latest.avg_page_complexity > 0.6:
                        inc = max(1, inc // 2)
                    current += inc
                    logger.debug(
                        "Throughput improving (%.2f > %.2f) — increasing batch by %d",
                        latest.throughput_pages_per_sec,
                        avg_prior,
                        inc,
                    )

        self._current_batch_size = self._clamp(current)

    def _clamp(self, value: int) -> int:
        """Clamp *value* to the configured [min, max] range."""
        return max(
            self._config.min_batch_size,
            min(value, self._config.max_batch_size),
        )
