# Pipeline Hardware Tuning Profiles

Tuning guide for the EDCOCR async producer-consumer pipeline. All values
are set via environment variables in `docker-compose.yml` or `coordinator/.env`.

## How the Pipeline Uses Resources

| Stage | Env Var | Default | Bound by |
|---|---|---|---|
| GPU OCR Workers | `NUM_WORKERS` | 12 | VRAM (~400-800 MB per worker) |
| CPU Extractors | `NUM_EXTRACTORS` | 8 | CPU cores, I/O bandwidth |
| Compressors | `NUM_COMPRESSORS` | 8 | CPU cores (Ghostscript) |
| Image Queue | `IMAGE_QUEUE_SIZE` | 200 | Host RAM (~20 MB per slot) |
| Extractor Mode | `EXTRACTOR_MODE` | thread | See below |

### EXTRACTOR_MODE

- **thread** (default): Uses Python threads. Best for typical workloads because
  extractors are I/O-bound (PDF rendering, disk reads) and threads avoid IPC overhead.
- **process**: Uses a `ProcessPoolExecutor`. Bypasses GIL but adds serialization cost.
  Useful only when extractors are CPU-bottlenecked with very large NUM_EXTRACTORS.
- **auto**: Resolves to `process` when `NUM_EXTRACTORS > 4`, else `thread`.

### IMAGE_QUEUE_SIZE and RAM

Each queued image consumes approximately 20 MB of host RAM (300 DPI, full-color).
A queue size of 200 requires roughly 4 GB of available RAM as buffer. If your host
is RAM-constrained, reduce this proportionally. Formula:

    IMAGE_QUEUE_SIZE = (available_ram_gb - 4) / 0.02

Reserve at least 4 GB for the OS, PaddleOCR models, and Python overhead.

### NUM_COMPRESSORS

Ghostscript compression is CPU-bound and single-threaded per document. Set this
to roughly `cpu_cores / 2` so compression does not starve the extractors. On hosts
with fewer than 8 cores, use 2-4.

## Pre-Built Profiles

### 8 GB VRAM (GTX 1080, RTX 3050, T4 budget)

```yaml
environment:
  - NUM_WORKERS=4
  - NUM_EXTRACTORS=4
  - NUM_COMPRESSORS=2
  - IMAGE_QUEUE_SIZE=50
```

Expected: ~2-3 PPM. VRAM is the hard constraint; exceeding 6 workers risks OOM.

### 12 GB VRAM (RTX 3060, RTX 4060)

```yaml
environment:
  - NUM_WORKERS=10
  - NUM_EXTRACTORS=6
  - NUM_COMPRESSORS=4
  - IMAGE_QUEUE_SIZE=150
```

Expected: ~4-6 PPM. Keep workers at or below 14 to avoid CUDA OOM.

### 16 GB VRAM (RTX 4070 Ti, T4, L4)

```yaml
environment:
  - NUM_WORKERS=16
  - NUM_EXTRACTORS=8
  - NUM_COMPRESSORS=8
  - IMAGE_QUEUE_SIZE=250
```

Expected: ~8-10 PPM. Datacenter cards (T4/L4) have lower memory bandwidth; reduce
NUM_WORKERS by 2-4 if GPU-Util stays high but throughput plateaus.

### 24 GB VRAM (RTX 3090 / 4090)

Full profile (32+ GB host RAM):
```yaml
environment:
  - NUM_WORKERS=24
  - NUM_EXTRACTORS=16
  - NUM_COMPRESSORS=12
  - IMAGE_QUEUE_SIZE=400
```

Expected: ~12-15 PPM.

RAM-constrained profile (16 GB host RAM, currently deployed in docker-compose.yml):
```yaml
environment:
  - NUM_WORKERS=8
  - NUM_EXTRACTORS=8
  - NUM_COMPRESSORS=4
  - IMAGE_QUEUE_SIZE=50
  - EXTRACTOR_MODE=thread
```

Expected: ~5-7 PPM. Keeps host RAM usage under 12 GB total.

### 48-80 GB VRAM (A100, H100)

```yaml
environment:
  - NUM_WORKERS=48
  - NUM_EXTRACTORS=32
  - NUM_COMPRESSORS=24
  - IMAGE_QUEUE_SIZE=1000
```

Expected: ~25-35 PPM. The bottleneck shifts to CPU extraction speed. Ensure the
host has 32+ cores and 64+ GB RAM to keep the image queue fed.

## Live Tuning Procedure

1. Run `watch -n 1 nvidia-smi` on the host while processing a batch.
2. Observe **GPU-Util** and **Memory-Usage**.
3. If GPU-Util < 70% and Memory < 80%, increase `NUM_WORKERS` by +2.
4. If you see `CUDA OutOfMemory` in logs, decrease `NUM_WORKERS` by -2.
5. If GPU is saturated but the monitor shows `Q_Img=0` frequently, increase
   `NUM_EXTRACTORS` to feed the pipeline faster.
6. Restart the container after changing environment variables.
