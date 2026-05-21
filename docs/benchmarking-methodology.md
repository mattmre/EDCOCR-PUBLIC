# Benchmarking Methodology

**Status**: Active
**Framework**: `benchmark_pipeline.py`
**Last Updated**: 2026-05-20

---

## Overview

The EDCOCR benchmarking framework measures pipeline performance before and after feature additions. Its primary purpose is to establish baseline metrics before (Document Intelligence) so that performance regressions can be detected and quantified.

---

## Metrics Collected

### Timing Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `total_duration_seconds` | seconds | Wall-clock time for the entire benchmark run |
| `pages_per_minute` | PPM | Overall throughput accounting for pipeline concurrency |
| `avg_time_per_page_ms` | ms | Mean total processing time per page (all stages) |
| `p50_time_per_page_ms` | ms | Median per-page time (50th percentile) |
| `p95_time_per_page_ms` | ms | 95th percentile per-page time |
| `p99_time_per_page_ms` | ms | 99th percentile per-page time |

### Stage Timings

Each page passes through four pipeline stages. Average time per stage:

| Stage | What It Measures |
|-------|-----------------|
| `extraction_avg_ms` | PDF rasterization to images at 300 DPI (CPU-bound) |
| `ocr_avg_ms` | PaddleOCR PP-OCRv4 inference (GPU-bound) |
| `assembly_avg_ms` | Page stitching into PDF + text file write |
| `compression_avg_ms` | Ghostscript PDF optimization |

### Memory Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `peak_memory_mb` | MB | Maximum resident set size during processing |
| `avg_memory_mb` | MB | Average memory usage across the run |

### Throughput

| Metric | Unit | Description |
|--------|------|-------------|
| `extraction_queue_throughput` | items/sec | Rate of pages entering the extraction stage |
| `ocr_queue_throughput` | items/sec | Rate of pages entering the OCR stage |
| `assembly_queue_throughput` | items/sec | Rate of pages entering the assembly stage |

---

## Modes

### Simulate Mode

```bash
python benchmark_pipeline.py --mode simulate --pages 100
```

**Purpose**: Test the benchmarking harness without GPU/PaddleOCR dependencies.

**How it works**:
1. For each simulated page, draws random timing values from uniform distributions based on documented pipeline characteristics
2. Computes wall-clock time by dividing stage work across the configured thread counts (e.g., OCR time divided by NUM_WORKERS)
3. Generates synthetic memory samples based on queue depth

**Timing distributions** (from production observations and roadmap estimates):

| Stage | Range (ms) | Source |
|-------|-----------|--------|
| Extraction | 30-80 | PDF rasterization at 300 DPI |
| OCR | 200-500 | PaddleOCR PP-OCRv4 documented range |
| Assembly | 10-40 | Page stitching + I/O |
| Compression | 50-200 | Ghostscript /prepress quality |

**Concurrency model**: The simulated wall-clock time accounts for parallelism:
- Extraction time / NUM_EXTRACTORS (8 threads)
- OCR time / NUM_WORKERS (12 threads)
- Assembly time / 1 (single assembler thread)
- Compression time / NUM_COMPRESSORS (8 threads)

**Limitations**:
- Does not capture real GPU behavior, thermal throttling, or memory pressure
- Uniform distributions are a simplification; real workloads have multi-modal distributions (text-heavy vs image-heavy pages)
- Does not model queue backpressure or thread contention

### Live Mode

```bash
python benchmark_pipeline.py --mode live --input-dir ocr_source/
```

**Purpose**: Instrument the actual OCR pipeline with timing wrappers.

**Current status**: Placeholder. The full implementation will:
1. Monkey-patch `queue.Queue.put` and `queue.Queue.get` with timing decorators
2. Run `OCR_GPU_Async.main` against the specified input directory
3. Collect per-page timing from instrumented queue operations
4. Aggregate metrics after pipeline completion
5. Collect real memory usage via `psutil`

**Requirements**: GPU, PaddleOCR, all production dependencies.

---

## Interpreting Results

### Pages Per Minute (PPM)

The primary throughput metric. Higher is better.

- **Baseline (OCR only)**: Established before - **With layout-only DocIntel**: Should be within 50% of baseline
- **With full DocIntel**: Should be within 100% of baseline

### Percentile Timings

- **P50**: Typical page processing time. Use for capacity planning.
- **P95**: Tail latency for most pages. Important for SLA compliance.
- **P99**: Worst-case latency (excluding outliers). Indicates pathological pages.

A large gap between P50 and P95/P99 suggests some pages are significantly harder to process (e.g., complex tables, many-language documents, or degraded scans).

### Stage Breakdown

If one stage dominates the total time, that stage is the bottleneck:
- **Extraction dominant**: Increase `NUM_EXTRACTORS` or use faster storage
- **OCR dominant**: Increase `NUM_WORKERS` (if VRAM allows) or reduce DPI
- **Assembly dominant**: Unlikely bottleneck; check I/O throughput
- **Compression dominant**: Increase `NUM_COMPRESSORS` or reduce quality setting

### Comparing Runs

```bash
python benchmark_pipeline.py --compare benchmark_results/run1.json benchmark_results/run2.json
```

The comparison report shows:
- Absolute values for both runs
- Percentage change with BETTER/WORSE indicators
- Pass/fail against performance targets

---

## Performance Targets

From the Document Intelligence Roadmap:

| Mode | Max Acceptable Slowdown | Justification |
|------|------------------------|---------------|
| Layout-only | < 50% | Layout detection adds ~20-50ms/page |
| Tables-only | < 100% | Table recognition adds ~100-300ms/page |
| Full DocIntel | < 100% | Full analysis adds ~200-500ms/page |
| GPU Memory increase | < 2GB per worker | Avoid OOM with 12 workers |

### Acceptance Criteria

A 1. PPM is within the slowdown threshold for the feature's mode
2. Peak memory does not exceed baseline + 2GB per worker
3. P99 latency does not exceed 3x the baseline P99
4. No queue starvation (all throughput metrics > 0)

---

## Standard Test Corpus

For reproducible benchmarks, use a standardized test corpus:

### Recommended Corpus Composition

| Category | Count | Description |
|----------|-------|-------------|
| Simple text PDFs | 30 | 1-5 pages, text-only, born-digital |
| Scanned documents | 30 | 1-10 pages, 300 DPI scans |
| Mixed content | 20 | Tables + text + images |
| Multi-language | 10 | Non-English documents |
| Large documents | 10 | 50+ pages each |

**Total**: 100 documents, ~500-1000 pages

### Corpus Requirements
- Documents should be representative of production workloads
- Include both born-digital and scanned PDFs
- Include a range of page counts (1 to 100+)
- No confidential or privileged content in benchmark corpus
- Store in a dedicated `benchmark_corpus/` directory (not checked into git)

---

## Results Storage

Results are stored as JSON files in `benchmark_results/`:

```
benchmark_results/
  benchmark_simulate_a1b2c3d4e5f6.json
  benchmark_simulate_f7g8h9i0j1k2.json
  benchmark_live_m3n4o5p6q7r8.json
```

Each file contains the full `BenchmarkMetrics` data (excluding raw per-page timings) for reproducibility and comparison.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-10 | Initial methodology document |
