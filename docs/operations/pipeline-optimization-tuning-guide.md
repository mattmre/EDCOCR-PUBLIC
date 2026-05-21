# Pipeline Optimization Tuning Guide

## 1. Overview

The EDCOCR pipeline includes five opt-in optimization modules that can be enabled independently to improve throughput, reduce latency, and optimize resource utilization. All five are disabled by default so the baseline pipeline behavior remains unchanged until an operator explicitly enables them.

| Module | Env Var | What It Does |
|--------|---------|--------------|
| **Adaptive Batch Sizing** | `ENABLE_ADAPTIVE_BATCH=true` | Dynamically adjusts chunk sizes based on document complexity and processing throughput |
| **Page Cache** | `ENABLE_PAGE_CACHE=true` | LRU cache that avoids re-processing identical pages across documents |
| **Page Routing** | `ENABLE_PAGE_ROUTING=true` | Routes individual pages to different OCR backends based on complexity analysis |
| **GPU Kernel Fusion** | `ENABLE_GPU_OPTIMIZATION=true` | Batches preprocessing operations (resize, normalize, pad) for GPU acceleration |
| **Benchmark Pipeline** | CLI tool (`benchmark_pipeline.py`) | Measures pipeline performance with simulated or live workloads for tuning feedback |

All modules integrate into `ocr_gpu_async.py` via lazy imports. If a module file is missing (e.g., in a stripped deployment), the pipeline logs a warning and continues without it.

---

## 2. Thread Tuning Reference Table

The production pipeline (`ocr_gpu_async.py`) runs 6 concurrent stages across 31 threads by default. Every thread count and queue size is configurable via environment variables with bounds-checked parsing.

### Tunable Constants

| Constant | Env Var | Default | Min | Max | Description |
|----------|---------|---------|-----|-----|-------------|
| `NUM_EXTRACTORS` | `NUM_EXTRACTORS` | 8 | 1 | 64 | CPU threads for PDF/image rasterization (I/O-bound) |
| `NUM_WORKERS` | `NUM_WORKERS` | 12 | 1 | 64 | GPU OCR worker threads (PaddleOCR inference) |
| `NUM_COMPRESSORS` | `NUM_COMPRESSORS` | 8 | 1 | 64 | Ghostscript PDF compression threads (CPU subprocess) |
| `IMAGE_QUEUE_SIZE` | `IMAGE_QUEUE_SIZE` | 200 | 1 | 10,000 | Buffer between extractors and OCR workers (RAM impact: ~20 MB per queued page) |
| `CHUNK_QUEUE_SIZE` | `CHUNK_QUEUE_SIZE` | 50 | 1 | 10,000 | Buffer between scheduler and extractors |
| `RESULT_QUEUE_SIZE` | `RESULT_QUEUE_SIZE` | 5,000 | 1 | 100,000 | Buffer between OCR workers and assembler |
| `COMPRESSION_QUEUE_SIZE` | `COMPRESSION_QUEUE_SIZE` | 5,000 | 1 | 100,000 | Buffer between assembler and compressors |
| `DPI` | `DPI` | 300 | 72 | 1,200 | Default scan resolution for OCR |
| `CHUNK_TARGET_SIZE` | `CHUNK_TARGET_SIZE` | 20 | 1 | 500 | Pages per extraction chunk |
| `PDF_CONVERSION_THREADS` | `PDF_CONVERSION_THREADS` | 1 | 1 | 16 | Threads within each pdf2image conversion call |
| `THREAD_JOIN_TIMEOUT` | `THREAD_JOIN_TIMEOUT` | 30 | 1 | 600 | Seconds to wait for thread shutdown |
| `EXTRACTOR_MODE` | `EXTRACTOR_MODE` | `thread` | -- | -- | `thread`, `process`, or `auto` (auto resolves to `process` when NUM_EXTRACTORS > 4) |

### Recommended Settings by VRAM Tier

| VRAM | NUM_WORKERS | NUM_EXTRACTORS | NUM_COMPRESSORS | IMAGE_QUEUE_SIZE | Notes |
|------|-------------|----------------|-----------------|------------------|-------|
| **8 GB** | 4-6 | 4 | 4 | 100 | Conservative; reduce workers to avoid VRAM OOM |
| **16 GB** | 8-12 | 8 | 8 | 200 | Default settings work well |
| **24 GB** | 12-16 | 8-10 | 8 | 200-300 | Increase workers to saturate GPU |
| **48 GB** | 16-24 | 10-12 | 8-10 | 300-400 | High throughput; monitor CPU saturation |

### Identifying the Bottleneck

Only 1 of 6 pipeline stages uses the GPU (the OCR workers). The remaining 5 stages are CPU and I/O bound. To identify which resource is the constraint:

**GPU-bound symptoms:**
- `Q_Img` (image queue) is consistently near zero (workers drain it faster than extractors fill it)
- GPU utilization at 95-100% (`nvidia-smi`)
- Increasing `NUM_WORKERS` does not improve throughput

**CPU-bound symptoms:**
- `Q_Img` is consistently full or near `IMAGE_QUEUE_SIZE` (workers cannot keep up)
- CPU usage maxed but GPU is idle between batches
- Increasing `NUM_EXTRACTORS` helps throughput

**I/O-bound symptoms:**
- Both queues low, but throughput is below expected
- High disk wait times (check `iostat` or `iotop`)
- Switching `EXTRACTOR_MODE=process` may help by parallelizing disk reads

**Tuning actions:**

| Bottleneck | What to Tune |
|------------|-------------|
| GPU-bound | Increase `NUM_WORKERS` (up to VRAM limit), enable GPU optimization, enable adaptive batch sizing |
| CPU-bound on extraction | Increase `NUM_EXTRACTORS`, set `EXTRACTOR_MODE=process`, increase `PDF_CONVERSION_THREADS` |
| CPU-bound on compression | Increase `NUM_COMPRESSORS` |
| I/O-bound | Increase `IMAGE_QUEUE_SIZE` to absorb bursts, use faster storage (NVMe), reduce `DPI` to 200 if accuracy permits |
| Memory pressure | Decrease `IMAGE_QUEUE_SIZE`, decrease `NUM_WORKERS`, reduce `DPI` |

---

## 3. Adaptive Batch Sizing

### What It Does

The `adaptive_batch.py` module dynamically adjusts the chunk size (number of pages per extraction batch) based on document complexity, memory pressure, and observed throughput. Instead of using a fixed `CHUNK_TARGET_SIZE` for every document, the sizer learns from processing history and adapts.

### How It Works

1. **Complexity scoring** -- Each page is scored on a 0.0-1.0 scale using weighted factors:
   - Pixel area relative to A4 at 300 DPI (weight: 0.3)
   - File size relative to 2 MB reference (weight: 0.3)
   - Table presence (weight: 0.2)
   - Embedded image presence (weight: 0.2)
   - DPI scaling factor for high-resolution scans

2. **Batch recommendation** -- Higher complexity pages get smaller batches (down to 20% of max). For example, a complex page with tables and images at 600 DPI might receive a batch size of 6 instead of the default 20.

3. **Adaptation loop** -- After a configurable warmup period (default: 3 batches), the sizer adjusts based on:
   - **Memory pressure**: Reduces batch size when peak memory exceeds target
   - **Throughput trend**: Increases batch size when throughput improves vs. historical average
   - **Complexity dampening**: Halves the increase step for high-complexity batches (score > 0.6)

### Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Enable | `ENABLE_ADAPTIVE_BATCH` | `false` | Master toggle |
| Strategy | `ADAPTIVE_BATCH_STRATEGY` | `adaptive` | `fixed`, `adaptive`, `memory_aware`, or `throughput_optimal` |
| Max batch size | `ADAPTIVE_BATCH_MAX` | `32` | Upper bound for batch size |
| Min batch size | (code default) | `1` | Lower bound for batch size |
| Target memory | (code default) | `75.0` | Target memory utilization percentage |
| Target latency | (code default) | `500.0` ms | Target per-batch latency |
| Warmup batches | (code default) | `3` | Batches before adaptation begins |
| Adjustment factor | (code default) | `0.1` | Fractional step size for adjustments |

### When to Enable

Enable adaptive batch sizing when:
- Your workload has **high variance in document complexity** (mix of simple text pages and complex scanned forms with tables)
- You observe memory spikes from large batch sizes on complex documents
- You want to automatically trade batch size for stability without manual tuning

Do not bother enabling if:
- All documents are similar in complexity (e.g., uniform typed correspondence)
- You have already tuned `CHUNK_TARGET_SIZE` to an optimal fixed value

### Expected Impact

- **Throughput**: 5-15% improvement on mixed-complexity workloads due to right-sized batches
- **Memory stability**: Reduced peak memory by avoiding oversized batches on complex pages
- **Tail latency**: Lower p95/p99 processing times for complex documents

---

## 4. Page Cache

### What It Does

The `page_cache.py` module provides a thread-safe LRU page cache that stores processed page results in memory. When the same page content is encountered again (keyed by a content hash), the cached result is returned immediately, skipping OCR entirely.

### How It Works

- Pages are keyed by content hash (typically SHA-256 of the page image bytes)
- Each entry stores raw page bytes plus optional metadata (OCR text, confidence)
- Eviction uses LRU (least recently used) when either byte limit or entry count limit is exceeded
- Optional per-entry TTL for time-based expiry
- The monitor thread logs cache hit rate statistics every reporting interval

### Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Enable | `ENABLE_PAGE_CACHE` | `false` | Master toggle |
| Max size (bytes) | `PAGE_CACHE_MAX_SIZE_BYTES` | `536870912` (512 MiB) | Maximum aggregate data stored |
| Max entries | `PAGE_CACHE_MAX_ENTRIES` | `1024` | Maximum number of cached pages |
| Default TTL | `PAGE_CACHE_DEFAULT_TTL` | `0` (no expiry) | Time-to-live in seconds |

### Memory Sizing Guide

Each cached page stores the full page image bytes. At 300 DPI, a typical A4 page image is 2-5 MB as a compressed PNG or 20-25 MB as raw RGB.

| Cache Size | Approximate Capacity (compressed) | RAM Overhead |
|------------|----------------------------------|-------------|
| 128 MiB | ~25-60 pages | Low |
| 256 MiB | ~50-120 pages | Moderate |
| 512 MiB (default) | ~100-250 pages | Moderate |
| 1 GiB | ~200-500 pages | High |
| 2 GiB | ~400-1000 pages | Very high -- ensure system has headroom |

**Sizing rule of thumb**: Set `PAGE_CACHE_MAX_SIZE_BYTES` to no more than 10-15% of total system RAM, leaving room for the OCR model (PaddleOCR uses ~1.5-3 GB VRAM), image queues, and OS buffers.

### When It Helps

- **Duplicate pages across documents** -- Common in legal discovery where cover sheets, footers, or boilerplate pages appear in many documents
- **Multi-tenant same-document requests** -- Multiple API jobs processing the same source file
- **Reprocessing after partial failure** -- Crash-resumed runs benefit from cached earlier pages

### When It Does Not Help

- Every page is unique (no duplicates in the workload)
- Working set exceeds cache size, causing constant eviction (hit rate near 0%)
- Memory is already tight -- the cache competes with image queues for RAM

### Monitoring Cache Performance

The monitor thread logs cache statistics every interval:

```
Cache: hits=142 misses=58 evictions=3 hit_rate=71.0%
```

A healthy cache has a hit rate above 30%. If evictions are high and hit rate is low, increase `PAGE_CACHE_MAX_SIZE_BYTES` or `PAGE_CACHE_MAX_ENTRIES`. If hit rate stays near 0%, your workload has no page duplication and the cache should be disabled to reclaim memory.

---

## 5. GPU Kernel Fusion

### What It Does

The `gpu_optimization.py` module batches image preprocessing operations (resize, normalize, pad) into fused GPU kernels instead of processing images one at a time on the CPU. It also probes GPU hardware capabilities and recommends optimal settings.

### How It Works

1. **Capability detection** -- On startup, `GpuOptimizer.detect_capabilities` queries CUDA devices for compute capability, total memory, FP16/INT8/tensor core support.

2. **Config recommendation** -- Based on detected hardware, the module recommends:
   - **Optimization level**: `none`, `basic`, or `aggressive` (based on compute capability)
   - **Fusion strategy**: `preprocess_batch`, `inference_batch`, or `full_pipeline` (based on VRAM)
   - **Max batch images**: Computed from available VRAM (50% of total divided by per-image estimate)
   - **Precision**: FP16 enabled for compute capability >= 5.3, INT8 for >= 6.1

3. **Batch preprocessing** -- The `BatchPreprocessor` groups images into sub-batches of up to `max_batch_images` and processes them through a unified resize-normalize-pad pipeline on the GPU.

### Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Enable | `ENABLE_GPU_OPTIMIZATION` | `false` | Master toggle |
| Optimization level | `GPU_OPTIMIZATION_LEVEL` | `auto` | `none`, `basic`, `aggressive`, `auto` |
| Fusion strategy | `GPU_FUSION_STRATEGY` | `none` | `none`, `preprocess_batch`, `inference_batch`, `full_pipeline` |
| Max batch images | `GPU_MAX_BATCH_IMAGES` | `8` | Maximum images per fused preprocessing batch |
| Enable FP16 | `GPU_ENABLE_FP16` | `false` | Half-precision inference (requires CC >= 5.3) |

### VRAM Overhead

The preprocessing batch consumes additional VRAM proportional to batch size:

| Batch Size | Image Size | Estimated VRAM (per batch) |
|-----------|-----------|---------------------------|
| 1 | 640x640x3 | ~6 MB |
| 4 | 640x640x3 | ~25 MB |
| 8 | 640x640x3 | ~50 MB |
| 16 | 640x640x3 | ~100 MB |
| 32 | 640x640x3 | ~200 MB |

These estimates include a 1.3x overhead multiplier for GPU workspace and memory fragmentation.

### Recommended Settings by GPU

| GPU | Compute Capability | Level | Strategy | Max Batch | FP16 |
|-----|-------------------|-------|----------|-----------|------|
| GTX 1060 (6 GB) | 6.1 | basic | preprocess_batch | 4 | Yes |
| RTX 2080 (8 GB) | 7.5 | aggressive | full_pipeline | 8 | Yes |
| T4 (16 GB) | 7.5 | aggressive | full_pipeline | 16 | Yes |
| A10G (24 GB) | 8.6 | aggressive | full_pipeline | 24 | Yes |
| A100 (40 GB) | 8.0 | aggressive | full_pipeline | 32 | Yes |

When `GPU_OPTIMIZATION_LEVEL=auto` (the default), the module auto-detects these values. Manual overrides are only needed for non-standard deployment scenarios.

### Expected Impact

- **Preprocessing latency**: 20-40% reduction from batched resize/normalize
- **GPU utilization**: Higher sustained utilization from reduced CPU-GPU transfer gaps
- **FP16 inference**: Up to 2x throughput on tensor-core GPUs (quality impact is negligible for OCR)

---

## 6. Page Routing

### What It Does

The `page_routing.py` module routes individual pages to different OCR backends based on a complexity analysis of each page's features. Instead of sending every page through the same PaddleOCR GPU path, pages are classified and dispatched to the most efficient backend.

### Available Routing Targets

| Target | Description | Typical Speed |
|--------|------------|--------------|
| `gpu_paddle` | PaddleOCR on GPU (default) | ~120 ms/page (base) |
| `gpu_tesseract` | Tesseract on GPU-equipped host | ~200 ms/page (base) |
| `cpu_paddle` | PaddleOCR on CPU | ~800 ms/page (base) |
| `cpu_tesseract` | Tesseract on CPU | ~600 ms/page (base) |
| `cpu_onnx` | PaddleOCR via ONNX Runtime on CPU | ~350 ms/page (base) |
| `skip` | Skip processing entirely | 0 ms |

Duration estimates are scaled by complexity score (0.5x for simple pages, 1.5x for complex pages).

### Built-in Routing Rules (Priority Order)

| Priority | Rule | Condition | Target |
|----------|------|-----------|--------|
| 100 | skip_tiny_pages | Width < 100 AND height < 100 | SKIP |
| 90 | handwritten_gpu | `is_handwritten` is true | GPU_PADDLE |
| 80 | tables_gpu | `has_tables` is true | GPU_PADDLE |
| 70 | high_complexity_gpu | Complexity score > 0.8 | GPU_PADDLE |
| 60 | low_complexity_onnx | Complexity score < 0.2 | CPU_ONNX |

Pages that match no rule fall through to the default target (`gpu_paddle`).

### Configuration

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Enable page routing | `ENABLE_PAGE_ROUTING` | `false` | Master toggle for complexity-based routing |

Custom rules can be added programmatically via `PageRouter.add_rule`.

### Related: Engine Selection

The `engine_selection.py` module provides a separate, simpler routing mechanism controlled by:

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Engine selection | `OCR_ENGINE_SELECTION` | `paddle` | `auto`, `paddle`, `tesseract`, `easyocr` |

In `auto` mode, each page image is analyzed for variance, edge density, and skew angle. Clean documents (high variance, low edge density, low skew) are routed to Tesseract for speed, while complex or degraded scans use PaddleOCR for accuracy.

### When Routing Helps vs. Hurts

**Helps throughput when:**
- Workload mixes simple typed text (route to fast CPU/ONNX path) with complex scans (keep on GPU)
- GPU is the bottleneck -- offloading simple pages to CPU paths frees GPU for hard pages
- Running mixed CPU+GPU worker deployment (distributed pipeline)

**Hurts throughput when:**
- All pages are similar complexity (routing overhead adds latency for no benefit)
- CPU paths are slower than GPU and there is available GPU headroom
- Feature extraction for routing decisions adds measurable overhead per page

### Monitoring Routing Decisions

The monitor thread logs routing statistics when enabled:

```
Routing: {'gpu_paddle': 45, 'cpu_onnx': 32, 'skip': 3}
```

If nearly all pages route to the same target, routing is adding overhead without benefit and should be disabled.

---

## 7. Benchmark Harness Usage

### Running Benchmarks

The `benchmark_pipeline.py` tool supports two modes:

**Simulate mode** (no GPU required, synthetic timing):
```bash
python benchmark_pipeline.py --mode simulate --pages 500
```

**Live mode** (requires full pipeline dependencies and source documents):
```bash
python benchmark_pipeline.py --mode live --input-dir /app/ocr_source
```

### Viewing Results

```bash
# Display all benchmark runs
python benchmark_pipeline.py --report

# Display the most recent run
python benchmark_pipeline.py --report latest

# Compare two runs side-by-side
python benchmark_pipeline.py --compare benchmark_results/run1.json benchmark_results/run2.json
```

### Understanding the Output

Each benchmark run produces a JSON file in `benchmark_results/` with these key metrics:

| Metric | What It Measures | What to Watch |
|--------|-----------------|---------------|
| `pages_per_minute` | Overall pipeline throughput | Primary tuning target |
| `avg_time_per_page_ms` | Mean per-page latency | Overall efficiency |
| `p95_time_per_page_ms` | 95th percentile latency | Tail latency (indicates outlier pages) |
| `p99_time_per_page_ms` | 99th percentile latency | Worst-case page latency |
| `peak_memory_mb` | Peak process memory | Watch for OOM risk |
| `extraction_avg_ms` | Avg time in extraction stage | CPU/I/O bottleneck indicator |
| `ocr_avg_ms` | Avg time in OCR stage | GPU bottleneck indicator |
| `assembly_avg_ms` | Avg time in assembly stage | Should be low (<50 ms) |
| `compression_avg_ms` | Avg time in compression stage | Ghostscript overhead |

### Using Results to Tune

1. **If `ocr_avg_ms` dominates**: GPU is the bottleneck. Enable GPU optimization, increase `NUM_WORKERS`, or enable page routing to offload simple pages to CPU paths.

2. **If `extraction_avg_ms` dominates**: CPU extraction is the bottleneck. Increase `NUM_EXTRACTORS`, switch `EXTRACTOR_MODE=process`, or increase `PDF_CONVERSION_THREADS`.

3. **If `compression_avg_ms` is high**: Increase `NUM_COMPRESSORS` or, if compression is not needed, disable it.

4. **If `p95` is much higher than `avg`**: A subset of pages is disproportionately slow. Enable adaptive batch sizing to handle complex pages more carefully, or enable DPI escalation threshold tuning.

5. **If `peak_memory_mb` is close to system limit**: Reduce `IMAGE_QUEUE_SIZE`, decrease `NUM_WORKERS`, or enable memory-aware adaptive batching.

### Comparison Analysis

The `--compare` output shows deltas with direction indicators:

```
  Metric                              Baseline                       Current
  -------------------------------------------------------------------
  Pages/Minute                          142.5        158.3 (+11.1% BETTER)
  Avg ms/page                           421.2        379.5 (-9.9% BETTER)
  Peak Memory (MB)                     1024.0       1180.0 (+15.2% WORSE)
```

Performance targets from the comparison report:
- **Layout-only threshold**: Within 50% slowdown is passing
- **Full DocIntel threshold**: Within 100% slowdown is passing

---

## 8. Tuning Playbook

Follow this step-by-step procedure to tune an EDCOCR deployment:

### Step 1: Establish a Baseline

Run a benchmark against your actual document workload:

```bash
# Place representative documents in ocr_source/
python benchmark_pipeline.py --mode live --input-dir /app/ocr_source
```

Record the baseline `pages_per_minute`, `p95_time_per_page_ms`, and `peak_memory_mb`.

If you cannot run live mode (no GPU), use simulate mode for relative comparisons:
```bash
python benchmark_pipeline.py --mode simulate --pages 1000
```

### Step 2: Identify the Bottleneck

Check the stage-level timing breakdown:

| If This Stage Dominates | Bottleneck Type | Next Step |
|------------------------|----------------|-----------|
| `extraction_avg_ms` > `ocr_avg_ms` | CPU / I/O extraction | Go to Step 3a |
| `ocr_avg_ms` > `extraction_avg_ms` | GPU OCR | Go to Step 3b |
| `compression_avg_ms` > 200 ms | Compression | Go to Step 3c |
| All stages balanced, low throughput | Queue starvation or memory | Go to Step 3d |

Also check queue depths in the monitor log output:
- `Q_Img` consistently full -> workers cannot keep up (GPU-bound)
- `Q_Img` consistently empty -> extractors cannot keep up (CPU-bound)
- `Q_Asm` consistently full -> assembler bottleneck (single-threaded, rare)

### Step 3a: CPU/I/O Extraction Tuning

```bash
# Increase extraction parallelism
export NUM_EXTRACTORS=12

# Enable process-based extraction (better for CPU-bound work)
export EXTRACTOR_MODE=process

# Increase pdf2image internal threads
export PDF_CONVERSION_THREADS=4

# Increase queue buffer to absorb bursts
export IMAGE_QUEUE_SIZE=300
```

Re-run benchmark and compare.

### Step 3b: GPU OCR Tuning

```bash
# Increase OCR worker threads (check VRAM with nvidia-smi)
export NUM_WORKERS=16

# Enable GPU kernel fusion for batch preprocessing
export ENABLE_GPU_OPTIMIZATION=true

# Enable adaptive batch sizing for mixed workloads
export ENABLE_ADAPTIVE_BATCH=true

# Enable page routing to offload simple pages to CPU
export ENABLE_PAGE_ROUTING=true
```

Re-run benchmark and compare.

### Step 3c: Compression Tuning

```bash
# Increase compression threads
export NUM_COMPRESSORS=12
```

If compression is not required for your use case, it can be disabled by commenting out the compression queue enqueue call in the assembler (code modification required).

### Step 3d: Memory and Queue Tuning

```bash
# Reduce queue sizes to lower memory pressure
export IMAGE_QUEUE_SIZE=100
export CHUNK_QUEUE_SIZE=25

# Enable memory-aware adaptive batching
export ENABLE_ADAPTIVE_BATCH=true
export ADAPTIVE_BATCH_STRATEGY=memory_aware

# Enable page cache if documents have duplicate pages
export ENABLE_PAGE_CACHE=true
export PAGE_CACHE_MAX_SIZE_BYTES=268435456  # 256 MiB
```

### Step 4: Validate the Improvement

```bash
# Run the same benchmark again
python benchmark_pipeline.py --mode live --input-dir /app/ocr_source

# Compare against baseline
python benchmark_pipeline.py --compare benchmark_results/baseline.json benchmark_results/latest.json
```

Check that:
- `pages_per_minute` improved (or at minimum did not regress)
- `peak_memory_mb` stayed within safe limits (leave 20% headroom)
- `p95_time_per_page_ms` did not worsen significantly

### Step 5: Iterate

Pipeline tuning is iterative. After applying one round of changes:
1. Re-measure to confirm improvement
2. Re-identify the bottleneck (it may shift)
3. Apply the next targeted change
4. Stop when throughput meets your target or marginal gains are below 5%

---

## 9. Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| GPU utilization at 100% but throughput is low | Too many workers competing for VRAM, causing context-switching overhead | Reduce `NUM_WORKERS` until GPU util drops to 85-95% |
| OOM killed by Docker | `IMAGE_QUEUE_SIZE` too large, or too many workers | Reduce `IMAGE_QUEUE_SIZE` to 100, reduce `NUM_WORKERS`, add `--shm-size=2g` to Docker |
| Throughput drops after enabling page cache | Working set exceeds cache, causing constant eviction overhead | Increase `PAGE_CACHE_MAX_SIZE_BYTES` or disable cache if hit rate < 5% |
| Adaptive batch keeps shrinking to 1 | Memory pressure feedback loop or incorrect memory baseline | Set `ADAPTIVE_BATCH_STRATEGY=fixed` temporarily, verify system memory is adequate |
| Page routing sends everything to GPU | All pages score above 0.2 complexity | Adjust default routing rules, or disable routing for uniform workloads |
| Page routing sends everything to CPU_ONNX | All pages score below 0.2 complexity | Workload is simple enough that routing is working correctly -- verify accuracy is acceptable on ONNX path |
| GPU optimization probe fails at startup | CUDA not available or torch not installed | Verify `nvidia-smi` works in container, check torch CUDA build |
| Benchmark live mode import error | PaddleOCR not installed | Use `--mode simulate` for testing without GPU dependencies |
| Extractor process pool fails to start | Windows or permission issue with multiprocessing | Set `EXTRACTOR_MODE=thread` to fall back to threading |
| Queue `Q_Asm` grows unbounded | Assembler thread is slower than workers (rare) | Check disk I/O on output volume; the assembler is single-threaded and I/O-bound |
| Compression stage very slow | Ghostscript processing large PDFs | Increase `NUM_COMPRESSORS`, verify Ghostscript binary is installed correctly |
| FP16 enabled but no speedup | GPU compute capability < 5.3 | Disable `GPU_ENABLE_FP16`; FP16 requires Maxwell Gen2+ |
| High p99 but low average | Occasional complex pages dominate tail latency | Enable `ENABLE_ADAPTIVE_BATCH` and `ENABLE_DPI_ESCALATION` to handle outliers gracefully |
| Cache hit rate is 0% | No duplicate pages in workload | Disable `ENABLE_PAGE_CACHE` to reclaim memory |

---

## Related Documentation

- [CPU vs GPU Deployment Guide](../cpu-vs-gpu-analysis.md) -- Detailed benchmarks and cost analysis for choosing deployment mode
- [Production Cutover Runbook](production-cutover-runbook.md) -- Step-by-step production deployment guide
- [Configuration Reference](../06-CONFIGURATION-REFERENCE.md) -- Full environment variable reference
- [System Blueprint](../00-SYSTEM-BLUEPRINT.md) -- Architecture overview of the 6-stage pipeline
