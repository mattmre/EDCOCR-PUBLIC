"""Production pipeline profiling framework for EDCOCR.

Provides per-stage timing, CPU time, peak RSS memory tracking, and
structured JSON output for ongoing performance monitoring and regression
detection.

Usage:
    from profiling import PipelineProfiler

    profiler = PipelineProfiler()
    with profiler.stage("extraction"):
        # extraction work ...
    with profiler.stage("ocr"):
        # OCR work ...

    report = profiler.report()
    profiler.save("profiling_results/run_001.json")
"""

import collections
import cProfile
import json
import logging
import os
import platform
import pstats
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional: psutil for RSS memory tracking (graceful fallback)
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from ocr_local.config.version import __version__
except ImportError:
    __version__ = "unknown"


def _get_rss_mb() -> float:
    """Return current process RSS in megabytes, or 0.0 if unavailable."""
    if not _HAS_PSUTIL:
        return 0.0
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _detect_gpu_availability() -> dict:
    """Probe for GPU frameworks and return availability info."""
    gpu_info: dict = {"gpu_available": False, "backends": []}

    # Check PaddlePaddle
    try:
        import paddle  # noqa: F401

        paddle_gpu = paddle.is_compiled_with_cuda()
        gpu_info["backends"].append(
            {"name": "paddle", "cuda": paddle_gpu}
        )
        if paddle_gpu:
            gpu_info["gpu_available"] = True
    except Exception:
        pass

    # Check PyTorch
    try:
        import torch  # noqa: F401

        torch_gpu = torch.cuda.is_available()
        gpu_info["backends"].append(
            {
                "name": "torch",
                "cuda": torch_gpu,
                "device_count": torch.cuda.device_count() if torch_gpu else 0,
            }
        )
        if torch_gpu:
            gpu_info["gpu_available"] = True
    except Exception:
        pass

    return gpu_info


def _collect_environment() -> dict:
    """Collect environment metadata for the profiling report."""
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pipeline_version": __version__,
    }

    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            env["total_memory_mb"] = round(vm.total / (1024 * 1024), 1)
        except Exception:
            pass

    gpu_info = _detect_gpu_availability()
    env["gpu"] = gpu_info

    return env


@dataclass
class StageMetrics:
    """Metrics captured for a single profiling stage."""

    name: str
    wall_time_seconds: float = 0.0
    cpu_time_seconds: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    start_timestamp: str = ""
    end_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wall_time_seconds": round(self.wall_time_seconds, 6),
            "cpu_time_seconds": round(self.cpu_time_seconds, 6),
            "rss_before_mb": round(self.rss_before_mb, 2),
            "rss_after_mb": round(self.rss_after_mb, 2),
            "rss_peak_mb": round(self.rss_peak_mb, 2),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }


class _StageContext:
    """Internal context for an active profiling stage.

    Runs a background memory-sampling thread during the stage to capture
    peak RSS between entry and exit.
    """

    def __init__(self, name: str, sample_interval: float = 0.1):
        self.name = name
        self._sample_interval = sample_interval
        self._wall_start: float = 0.0
        self._cpu_start: float = 0.0
        self._rss_before: float = 0.0
        self._rss_samples: list[float] = []
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None

    def _memory_sampler(self) -> None:
        """Background thread that samples RSS at regular intervals."""
        while not self._stop_event.is_set():
            rss = _get_rss_mb()
            if rss > 0:
                self._rss_samples.append(rss)
            self._stop_event.wait(self._sample_interval)

    def start(self) -> None:
        self._rss_before = _get_rss_mb()
        if self._rss_before > 0:
            self._rss_samples.append(self._rss_before)
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()

        # Start background memory sampler
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(
            target=self._memory_sampler, daemon=True
        )
        self._sampler_thread.start()

    def stop(self) -> StageMetrics:
        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        # Stop memory sampler
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)

        # Final RSS sample
        rss_after = _get_rss_mb()
        if rss_after > 0:
            self._rss_samples.append(rss_after)

        rss_peak = max(self._rss_samples) if self._rss_samples else 0.0

        return StageMetrics(
            name=self.name,
            wall_time_seconds=wall_end - self._wall_start,
            cpu_time_seconds=cpu_end - self._cpu_start,
            rss_before_mb=self._rss_before,
            rss_after_mb=rss_after,
            rss_peak_mb=rss_peak,
            start_timestamp=datetime.fromtimestamp(
                time.time() - (wall_end - self._wall_start), tz=timezone.utc
            ).isoformat(),
            end_timestamp=datetime.now(timezone.utc).isoformat(),
        )


class PipelineProfiler:
    """Production profiling harness for the OCR pipeline.

    Thread-safe: multiple stages can run concurrently from different
    threads; each stage() context is independent.

    Example::

        profiler = PipelineProfiler()
        with profiler.stage("extraction"):
            run_extraction()
        with profiler.stage("ocr"):
            run_ocr()

        report = profiler.report()
        profiler.save("/tmp/profile.json")
    """

    def __init__(self, *, memory_sample_interval: float = 0.1):
        self._stages: list[StageMetrics] = []
        self._lock = threading.Lock()
        self._overall_wall_start: float | None = None
        self._overall_cpu_start: float | None = None
        self._overall_rss_peak: float = 0.0
        self._memory_sample_interval = memory_sample_interval
        self._environment: dict | None = None

    @contextmanager
    def stage(self, name: str):
        """Context manager that profiles a named pipeline stage.

        Multiple stages can overlap (run concurrently from different
        threads).  Each stage independently tracks wall time, CPU time,
        and peak RSS.
        """
        # Lazily start overall timing on first stage entry
        with self._lock:
            if self._overall_wall_start is None:
                self._overall_wall_start = time.perf_counter()
                self._overall_cpu_start = time.process_time()

        ctx = _StageContext(
            name, sample_interval=self._memory_sample_interval
        )
        ctx.start()
        try:
            yield
        finally:
            metrics = ctx.stop()
            with self._lock:
                self._stages.append(metrics)
                if metrics.rss_peak_mb > self._overall_rss_peak:
                    self._overall_rss_peak = metrics.rss_peak_mb

    def report(self) -> dict:
        """Return a structured profiling report as a dictionary.

        The report includes:
        - ``environment``: system/platform/GPU metadata
        - ``stages``: list of per-stage metric dicts
        - ``overall``: aggregate wall time, CPU time, peak memory
        - ``timestamp``: ISO-8601 generation timestamp
        """
        with self._lock:
            stages_snapshot = list(self._stages)
            wall_start = self._overall_wall_start
            cpu_start = self._overall_cpu_start
            peak_rss = self._overall_rss_peak

        # Compute overall metrics
        now_wall = time.perf_counter()
        now_cpu = time.process_time()

        total_wall = (now_wall - wall_start) if wall_start is not None else 0.0
        total_cpu = (now_cpu - cpu_start) if cpu_start is not None else 0.0

        # Sum of per-stage times (may exceed total_wall if stages overlap)
        stage_wall_sum = sum(s.wall_time_seconds for s in stages_snapshot)
        stage_cpu_sum = sum(s.cpu_time_seconds for s in stages_snapshot)

        # Collect environment lazily (once)
        if self._environment is None:
            self._environment = _collect_environment()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self._environment,
            "stages": [s.to_dict() for s in stages_snapshot],
            "overall": {
                "total_wall_time_seconds": round(total_wall, 6),
                "total_cpu_time_seconds": round(total_cpu, 6),
                "stage_wall_sum_seconds": round(stage_wall_sum, 6),
                "stage_cpu_sum_seconds": round(stage_cpu_sum, 6),
                "peak_rss_mb": round(peak_rss, 2),
                "stage_count": len(stages_snapshot),
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write the profiling report to a JSON file.

        Creates parent directories if needed.

        Args:
            path: Destination file path.

        Returns:
            Resolved Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.report()
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Profiling report saved to %s", path)
        return path.resolve()

    def reset(self) -> None:
        """Clear all collected stage data and timers."""
        with self._lock:
            self._stages.clear()
            self._overall_wall_start = None
            self._overall_cpu_start = None
            self._overall_rss_peak = 0.0
            self._environment = None


class FlameGraphProfiler:
    """cProfile-based profiler that captures function-level timing data.

    Can be used as a context manager or via explicit ``start()``/``stop()``
    calls.  Writes ``.prof`` files compatible with standard Python profiling
    tools (snakeviz, gprof2dot, etc.).

    Example::

        with FlameGraphProfiler() as fp:
            heavy_computation()
        print(fp.get_top_functions(10))
        fp.dump_stats("profile.prof")
    """

    def __init__(self):
        self._profile = cProfile.Profile()
        self._running = False

    # -- context manager ----------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Enable the cProfile profiler."""
        if not self._running:
            self._profile.enable()
            self._running = True

    def stop(self) -> None:
        """Disable the cProfile profiler."""
        if self._running:
            self._profile.disable()
            self._running = False

    # -- output -------------------------------------------------------------

    def dump_stats(self, filepath: str | Path) -> Path:
        """Save cProfile stats to a ``.prof`` file.

        Creates parent directories if needed.

        Args:
            filepath: Destination path (should end in ``.prof``).

        Returns:
            Resolved Path to the written file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._profile.dump_stats(str(filepath))
        return filepath.resolve()

    def get_top_functions(self, n: int = 20) -> list[tuple[str, float, int]]:
        """Return the top *n* functions sorted by cumulative time.

        Each entry is ``(function_name, cumulative_time, call_count)``.
        """
        stats = pstats.Stats(self._profile)
        # stats.stats maps (file, line, name) -> (cc, nc, tt, ct, callers)
        entries: list[tuple[str, float, int]] = []
        for (file, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
            func_name = f"{file}:{line}({name})"
            entries.append((func_name, ct, cc))

        # Sort by cumulative time descending
        entries.sort(key=lambda e: e[1], reverse=True)
        return entries[:n]

    def generate_report(self) -> dict:
        """Return a summary dict with top functions, total time, and call count."""
        top = self.get_top_functions(20)
        total_time = sum(ct for _, ct, _ in top) if top else 0.0
        total_calls = sum(cc for _, _, cc in top) if top else 0
        return {
            "top_functions": [
                {"function": fn, "cumulative_time": round(ct, 6), "call_count": cc}
                for fn, ct, cc in top
            ],
            "total_time": round(total_time, 6),
            "total_calls": total_calls,
        }


class ContinuousProfiler:
    """Background daemon that samples system metrics at regular intervals.

    Collects memory (MB), CPU percent, and optional queue depths in a
    rolling window (default: 1 hour at 5-second intervals = 720 samples).

    Gated by the ``CONTINUOUS_PROFILING_ENABLED`` environment variable:
    ``start()`` is a no-op unless ``CONTINUOUS_PROFILING_ENABLED=true``.

    Example::

        cp = ContinuousProfiler(interval=5)
        cp.start()
        # ... long-running work ...
        summary = cp.get_summary()
        cp.flush_to_file("metrics.jsonl")
        cp.stop()
    """

    def __init__(
        self,
        interval: float = 5.0,
        maxlen: int = 720,
        overhead_cap_ms: float = 50.0,
    ):
        self._interval = interval
        self._maxlen = maxlen
        self._overhead_cap_ms = overhead_cap_ms
        self._samples: collections.deque = collections.deque(maxlen=maxlen)
        self._queues: dict[str, object] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    # -- queue tracking -----------------------------------------------------

    def register_queue(self, name: str, q: object) -> None:
        """Register a queue whose ``qsize()`` will be sampled each tick."""
        with self._lock:
            self._queues[name] = q

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread.

        No-op unless ``CONTINUOUS_PROFILING_ENABLED=true`` in the
        environment.
        """
        if os.environ.get("CONTINUOUS_PROFILING_ENABLED", "").lower() != "true":
            return
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sampling thread."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
        self._running = False

    # -- internal -----------------------------------------------------------

    def _run(self) -> None:
        interval = self._interval
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            snapshot = self._take_snapshot()
            with self._lock:
                self._samples.append(snapshot)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Adaptive interval: double if sampling overhead exceeds cap
            if elapsed_ms > self._overhead_cap_ms:
                interval = min(interval * 2, 3600)
            self._stop_event.wait(interval)

    def _take_snapshot(self) -> dict:
        snapshot: dict = {"timestamp": time.time()}
        if _HAS_PSUTIL:
            try:
                proc = psutil.Process(os.getpid())
                snapshot["memory_mb"] = round(
                    proc.memory_info().rss / (1024 * 1024), 2
                )
                snapshot["cpu_percent"] = proc.cpu_percent(interval=0)
            except Exception:
                snapshot["memory_mb"] = 0.0
                snapshot["cpu_percent"] = 0.0
        else:
            snapshot["memory_mb"] = 0.0
            snapshot["cpu_percent"] = 0.0

        # Queue depths
        with self._lock:
            for qname, q in self._queues.items():
                try:
                    snapshot[f"queue_{qname}"] = q.qsize()
                except Exception:
                    snapshot[f"queue_{qname}"] = 0
        return snapshot

    # -- output -------------------------------------------------------------

    def flush_to_file(self, filepath: str | Path) -> Path:
        """Append current samples as JSONL to *filepath*.

        Creates parent directories if needed.  Each line is a JSON object
        representing one snapshot.  The internal buffer is cleared after
        flushing.

        Returns:
            Resolved Path to the written file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            samples = list(self._samples)
            self._samples.clear()
        with open(filepath, "a", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s, default=str) + "\n")
        return filepath.resolve()

    def get_summary(self) -> dict:
        """Return aggregate statistics (avg / max / p95) for each metric."""
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {}

        keys = [k for k in samples[0] if k != "timestamp"]
        summary: dict = {}
        for key in keys:
            vals = [s.get(key, 0) for s in samples if isinstance(s.get(key), (int, float))]
            if not vals:
                continue
            sorted_vals = sorted(vals)
            p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1)
            summary[key] = {
                "avg": round(sum(vals) / len(vals), 2),
                "max": round(max(vals), 2),
                "p95": round(sorted_vals[p95_idx], 2),
            }
        return summary
