# CPU vs GPU OCR Deployment Guide

## Executive Summary

Choosing between CPU and GPU deployment for the EDCOCR pipeline is primarily a question of **volume, budget, and operational constraints** -- not accuracy. Both modes run the same PaddleOCR PP-OCRv4 models and produce identical OCR output.

Key takeaways:

- **Most users and use cases favor CPU-only swarms** for cost efficiency, simpler operations, and easier horizontal scaling.
- **GPU excels at high-volume (50K+ pages/day) continuous processing** where sustained throughput and latency matter.
- **With ONNX Runtime optimization, CPU workers are 4-5x faster** than default PaddlePaddle CPU inference, closing the gap significantly.
- **The pipeline supports both modes transparently.** No code changes are required to switch between CPU and GPU -- only infrastructure configuration differs.

---

## Pipeline Architecture: What Uses GPU vs CPU

The production pipeline (`ocr_gpu_async.py`) runs 6 concurrent stages across 31 threads. Understanding which stages benefit from GPU acceleration is critical to making an informed deployment decision.

| Stage | Threads | Compute Type | Uses GPU? | Notes |
|-------|---------|-------------|-----------|-------|
| **Scheduler** (file scan) | 1 | I/O | No | FastText language detection (CPU, <1ms per document) |
| **CPU Extractors** (PDF to images) | 8 | I/O + CPU | No | Poppler rendering at 300 DPI via `pdf2image` |
| **OCR Workers** | 12 | GPU or CPU | **Only stage using GPU** | PaddleOCR inference (detection + recognition) |
| **Assembler** | 1 | CPU + I/O | No | PDF merge, NER, classification, extraction |
| **Compressors** | 8 | CPU (subprocess) | No | Ghostscript PDF optimization |
| **Monitor** | 1 (daemon) | Timer | No | Metrics collection, heartbeat file |

**Key insight: Only 1 of 6 pipeline stages uses GPU. Out of 31 threads, only 12 OCR worker threads can benefit from GPU acceleration.** The remaining 19 threads are exclusively CPU and I/O bound regardless of hardware.

This means that even in a GPU deployment, the majority of pipeline activity is CPU-based. Overprovisioning GPU resources while starving CPU resources is a common misconfiguration.

---

## Performance Benchmarks

### Per-Page OCR Inference Time

These benchmarks represent end-to-end OCR processing time per page (text detection + recognition + orientation classification) for a typical mixed-content business document at 300 DPI.

| Configuration | Time/Page | Pages/Min | Relative Speed |
|---------------|-----------|-----------|----------------|
| GPU (NVIDIA T4, 16 GB) | 200-400ms | 150-300 | 1.0x (baseline) |
| GPU (NVIDIA A10G, 24 GB) | 150-300ms | 200-400 | 1.3x |
| CPU (PaddlePaddle native) | 1.5-4.0s | 15-40 | 0.1-0.2x |
| CPU (ONNX Runtime) | 400ms-1.0s | 60-100 | 0.3-0.5x |
| CPU (OpenVINO, Intel Xeon) | 300-800ms | 75-130 | 0.4-0.7x |
| Tesseract 5 (CPU, clean docs) | 0.8-2.5s | 24-75 | 0.15-0.4x |

Notes:
- Times vary with page complexity (text density, number of text regions, image quality).
- GPU times assume warm inference (models already loaded). First-page cold start adds 2-5 seconds.
- Tesseract performance is highly dependent on document quality -- clean scans approach PaddleOCR speed.

### PaddleOCR Model Breakdown (PP-OCRv4)

PaddleOCR runs three models per page. Here is where the time is spent:

| Model | GPU (ms) | CPU (ms) | CPU:GPU Ratio |
|-------|-------:|-------:|--------------:|
| Text Detection (DB) | 98-128 | 490-586 | ~5x slower |
| Text Recognition (SVTR) | 2.5-8.8 | 37 | ~4-15x slower |
| Line Orientation | ~5 | ~20 | ~4x slower |
| **End-to-end per page** | **200-400** | **1500-4000** | **4-10x slower** |

The text detection model (DB) dominates processing time on both CPU and GPU. Recognition time scales with the number of detected text lines -- dense documents take proportionally longer.

### CPU Inference Optimization Impact

The default PaddlePaddle CPU backend leaves significant performance on the table. The following optimizations can dramatically improve CPU throughput:

| Backend | Relative CPU Speed | Notes |
|---------|-------------------|-------|
| PaddlePaddle native | 1.0x (baseline) | Default, no optimization |
| PaddlePaddle + MKL-DNN | ~1.5-2.0x | Intel CPUs, enabled by default in PaddleOCR 3.0.3+ |
| ONNX Runtime | **4-5x** | Best cross-platform option, works on AMD/ARM/Intel |
| OpenVINO (Intel) | **5-7x** | Intel CPUs only, best CPU performance |

With ONNX Runtime, a 4-vCPU instance processes 60-100 pages/minute -- competitive with older GPU instances for many workloads.

---

## Cost Analysis

### Cloud Instance Pricing (US-East, on-demand, 2025)

| Instance | Provider | vCPUs | RAM | GPU | $/Hour |
|----------|----------|------:|----:|-----|-------:|
| c5.xlarge | AWS | 4 | 8 GiB | None | $0.170 |
| c5.2xlarge | AWS | 8 | 16 GiB | None | $0.340 |
| c5.4xlarge | AWS | 16 | 32 GiB | None | $0.680 |
| g4dn.xlarge | AWS | 4 | 16 GiB | 1x T4 | $0.526 |
| g5.xlarge | AWS | 4 | 16 GiB | 1x A10G | $1.006 |
| NC4as T4 v3 | Azure | 4 | 28 GiB | 1x T4 | $0.526 |
| n1-standard-4 + T4 | GCP | 4 | 15 GiB | 1x T4 | ~$0.350 |

### Spot/Preemptible Pricing

Spot instances are a strong fit for OCR workloads because the pipeline has **page-level crash resume** -- a spot termination loses at most a few seconds of in-progress work.

| Instance | Provider | Spot $/Hour | Savings vs On-Demand |
|----------|----------|------------:|----------------------|
| g4dn.xlarge | AWS | ~$0.21 | 60% |
| T4 | GCP | ~$0.14 | 60% |
| c5.xlarge | AWS | ~$0.07 | 59% |

### Cost Per 1,000 Pages

This is the metric that matters for budget planning. It combines instance cost with throughput capacity.

| Configuration | Hourly Cost | Pages/Hr | Cost/1K Pages |
|---------------|------------:|---------:|--------------:|
| 1x GPU (T4, on-demand) | $0.53 | 9,000-18,000 | **$0.03-0.06** |
| 1x GPU (T4, spot) | $0.21 | 9,000-18,000 | **$0.01-0.02** |
| 4x CPU (PaddlePaddle native) | $0.68 | 960-2,400 | $0.28-0.71 |
| 4x CPU (ONNX Runtime) | $0.68 | 3,840-9,600 | **$0.07-0.18** |
| 4x CPU (ONNX, spot) | $0.28 | 3,840-9,600 | **$0.03-0.07** |
| 4x CPU (OpenVINO, Intel) | $0.68 | 4,480-12,480 | **$0.05-0.15** |

Key observations:
- GPU on-demand is the cheapest per-page option at scale.
- CPU with ONNX Runtime on spot instances is cost-competitive with GPU on-demand.
- Unoptimized CPU (PaddlePaddle native) is 5-10x more expensive per page than GPU.

### Break-Even Scenarios

#### Small workload: 10,000 pages/day

```
GPU: 1x g4dn.xlarge = $12.67/day (5.6% utilization -- wasteful)
CPU: 3x c5.xlarge (ONNX) = $12.24/day (92% utilization -- efficient)
CPU spot: 3x c5.xlarge spot = $4.80/day

Verdict: CPU swarm wins. GPU sits idle 94% of the time.
```

#### Medium workload: 50,000 pages/day

```
GPU: 1x g4dn.xlarge = $12.67/day (28% utilization)
CPU: 8x c5.xlarge (ONNX) = $32.64/day (78% utilization)
CPU spot: 8x c5.xlarge spot = $13.44/day

Verdict: GPU wins on on-demand pricing. CPU spot is competitive.
```

#### Large workload: 200,000 pages/day

```
GPU: 2x g4dn.xlarge = $25.34/day (56% utilization)
CPU: 25x c5.xlarge (ONNX) = $102/day
CPU spot: 25x c5.xlarge spot = $42/day

Verdict: GPU wins decisively. Managing 25 CPU nodes adds operational overhead.
```

### Decision Matrix

| Factor | Favors GPU | Favors CPU |
|--------|:---------:|:----------:|
| Volume > 50K pages/day | Yes | |
| Volume < 50K pages/day | | Yes |
| Spot/preemptible available | | Yes |
| Air-gapped / on-premises | | Yes (no CUDA drivers needed) |
| Latency-sensitive (real-time) | Yes | |
| Batch processing (overnight) | | Yes |
| Budget-constrained | | Yes |
| GPU nodes readily available | Yes | |
| Horizontal scaling preferred | | Yes |
| Mixed workload (OCR + other) | | Yes (no GPU waste) |
| Degraded/complex documents | Yes (faster re-processing) | |
| Clean scanned documents | | Yes (Tesseract competitive) |
| Minimal ops team | | Yes (simpler infrastructure) |
| Kubernetes native | | Yes (easier autoscaling) |

---

## Deployment Configurations

### CPU-Only Deployment (Recommended for most users)

The CPU-only deployment eliminates CUDA dependencies entirely. It is simpler to build, deploy, and scale.

**Minimum viable resources per worker:** 6 CPU cores, 10 GiB RAM.

Characteristics:
- No `nvidia.com/gpu` resource requests in Kubernetes
- ONNX Runtime inference backend enabled for 4-5x CPU speedup
- Higher worker concurrency per node (8-16 workers vs 2-4 for GPU)
- KEDA autoscaling tuned for CPU: scale-to-zero on idle, faster cooldown
- Smaller Docker images (no CUDA runtime, ~2-3 GB smaller)
- Simpler CI/CD (no GPU runners needed for testing)

Example Helm values override:

```yaml
# values-cpu-only.yaml
gpuWorker:
  replicas: 0              # Disable GPU workers entirely

cpuWorker:
  replicas: 3
  concurrency: 8
  queues: "ocr_cpu,cpu_general"  # Subscribe to CPU OCR queue
  env:
    OCR_INFERENCE_BACKEND: "onnx"
  resources:
    requests:
      cpu: "4"
      memory: 4Gi
    limits:
      cpu: "8"
      memory: 8Gi
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 20
    queueTarget: 5         # Scale up when queue depth > 5
    cooldownPeriod: 60      # Scale down after 60s idle
```

### GPU Deployment (High-volume production)

For sustained high-volume workloads where throughput per dollar matters most.

```yaml
# values-gpu.yaml (default)
gpuWorker:
  replicas: 2
  concurrency: 4           # Lower concurrency -- GPU handles parallelism
  queues: "ocr_gpu,cpu_general"
  resources:
    requests:
      nvidia.com/gpu: 1
      cpu: "4"
      memory: 8Gi
    limits:
      nvidia.com/gpu: 1
      cpu: "8"
      memory: 16Gi

cpuWorker:
  replicas: 1              # Offload compression, NER, classification
  queues: "cpu_general"
  resources:
    requests:
      cpu: "4"
      memory: 4Gi
```

### Hybrid Deployment (Best of both worlds)

Route documents based on complexity: GPU handles degraded and complex documents, CPU handles clean scans.

```yaml
# values-hybrid.yaml
gpuWorker:
  replicas: 1              # One GPU for complex/degraded docs
  queues: "ocr_gpu"        # GPU-only OCR queue
  resources:
    requests:
      nvidia.com/gpu: 1
      cpu: "4"
      memory: 8Gi

cpuWorker:
  replicas: 4              # CPU swarm for clean docs
  queues: "ocr_cpu,cpu_general"
  env:
    OCR_INFERENCE_BACKEND: "onnx"
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 15
    queueTarget: 5
```

This approach maximizes cost efficiency by reserving expensive GPU resources for documents that benefit most from them, while handling the bulk of clean scans on cheaper CPU workers.

---

## CPU Optimization Options

### ONNX Runtime (Recommended)

ONNX Runtime provides the best balance of performance and portability for CPU inference.

**How to enable:**
- Set `OCR_INFERENCE_BACKEND=onnx` as an environment variable.
- Models are auto-converted from PaddlePaddle format on first run. Subsequent runs use cached ONNX models.
- Provides 4-5x speedup over native PaddlePaddle CPU inference.
- Cross-platform: works on AMD, ARM (Apple Silicon, Graviton), and Intel processors.

**Installation:**
```bash
pip install onnxruntime        # CPU-only
pip install onnxruntime-gpu    # GPU + CPU (if needed)
```

**Trade-offs:**
- First-run model conversion takes 30-60 seconds per model (one-time cost).
- ONNX models consume additional disk space (~200 MB for PP-OCRv4 full model set).
- Some edge-case PaddlePaddle operators may not convert cleanly -- test with your document corpus.

### OpenVINO (Intel CPUs)

For Intel-specific deployments, OpenVINO provides the best CPU inference performance.

**How to enable:**
- Set `OCR_INFERENCE_BACKEND=openvino`.
- Provides an additional 10-30% speedup over ONNX Runtime on Intel Xeon and Core processors.
- Requires Intel hardware -- will not run on AMD or ARM.

**Installation:**
```bash
pip install openvino
```

### MKL-DNN (Default on Intel)

MKL-DNN (now oneDNN) is enabled automatically on Intel CPUs when using PaddlePaddle.

- Provides 1.5-2x speedup over the unoptimized baseline.
- No configuration needed -- PaddleOCR 3.0.3+ enables it by default.
- This is the "free" optimization you get without any changes.

### Optimization Comparison Summary

| Backend | Speedup | Platform | Effort |
|---------|---------|----------|--------|
| MKL-DNN (default) | 1.5-2x | Intel only | None (automatic) |
| ONNX Runtime | 4-5x | Any CPU | Set one env var |
| OpenVINO | 5-7x | Intel only | Set one env var + install package |

---

## Smart Engine Selection

The pipeline can route documents to different OCR engines based on document quality, reducing GPU usage for documents that do not need it.

### Quality-Based Routing

- **Pre-scan**: Each page is analyzed for image variance, text density, and degradation markers before OCR.
- **Clean documents** (high contrast, low noise): Routed to Tesseract, which is fast on CPU and highly accurate for clean English text.
- **Complex/degraded documents** (low contrast, noise, skew, non-Latin scripts): Routed to PaddleOCR, which handles these cases with significantly higher accuracy.

### Configuration

```bash
# Automatic engine selection (default in hybrid mode)
OCR_ENGINE_SELECTION=auto

# Force PaddleOCR for all documents
OCR_ENGINE_SELECTION=paddle

# Force Tesseract for all documents (fastest CPU option for clean docs)
OCR_ENGINE_SELECTION=tesseract
```

In `auto` mode, the routing decision is made per-page, so a single document can use both engines for different pages. The assembler merges results transparently.

---

## Air-Gapped CPU Deployment

CPU-only air-gapped deployments are simpler and smaller than GPU equivalents.

| Component | GPU Bundle Size | CPU Bundle Size |
|-----------|---------------:|----------------:|
| Base Docker image | ~8 GB | ~5 GB |
| CUDA runtime | ~2.5 GB | 0 (not needed) |
| PaddleOCR models (27 languages) | ~1.2 GB | ~1.2 GB (identical) |
| FastText language model | ~126 MB | ~126 MB (identical) |
| **Total** | **~12 GB** | **~6.5 GB** |

The 27 pre-downloaded language models are **identical for CPU and GPU** -- PaddleOCR models are not GPU-specific. The same model files work on both backends. The size difference comes entirely from eliminating the CUDA runtime and GPU-specific PaddlePaddle packages.

Air-gapped bundle scripts:
```bash
# Bundle for air-gapped deployment
scripts/airgap-bundle.sh

# Deploy on isolated network
scripts/airgap-deploy.sh
```

---

## Scaling Considerations

### CPU Scaling

CPU workers scale horizontally with near-linear throughput gains up to the point where the message broker or shared storage becomes a bottleneck.

| CPU Workers | Approx Pages/Hr (ONNX) | Notes |
|------------:|------------------------:|-------|
| 1 | 960-1,500 | Minimum viable |
| 4 | 3,840-6,000 | Small team workload |
| 8 | 7,680-12,000 | Medium organization |
| 16 | 15,000-24,000 | Large organization |
| 32 | 28,000-45,000 | Enterprise, approaching GPU territory |

### GPU Scaling

GPU scaling is limited by GPU availability and cost. Each GPU worker handles higher throughput but at a higher per-unit cost.

| GPU Workers (T4) | Approx Pages/Hr | Notes |
|------------------:|-----------------:|-------|
| 1 | 9,000-18,000 | Single-server deployment |
| 2 | 18,000-36,000 | Most production workloads |
| 4 | 36,000-72,000 | Large enterprise |
| 8 | 72,000-144,000 | Maximum practical scale |

### Bottleneck Analysis

At high scale, the OCR inference stage is rarely the bottleneck. Other stages that may limit throughput:

| Bottleneck | Symptom | Mitigation |
|------------|---------|------------|
| PDF extraction (Poppler) | Extraction queue fills up | Increase `NUM_EXTRACTORS`, use `EXTRACTOR_MODE=process` |
| Ghostscript compression | Compression queue grows | Increase `NUM_COMPRESSORS`, or disable compression |
| Assembler (single-threaded) | Result queue backs up | Minimize sidecar features (NER, classification, extraction) |
| Network I/O (NFS/S3) | All queues slow | Use local SSD for temp storage, async uploads |
| Message broker (RabbitMQ) | Task delivery stalls | Scale broker, increase prefetch |

---

## Frequently Asked Questions

**Q: Is PaddleOCR more accurate than Tesseract?**

A: Yes, significantly for complex layouts, degraded scans, handwritten text, and non-Latin scripts. For clean, well-scanned English documents, both achieve >95% character accuracy. The accuracy gap widens as document quality decreases -- PaddleOCR maintains >90% accuracy on documents where Tesseract drops below 80%.

**Q: Can I switch between CPU and GPU without code changes?**

A: Yes. The pipeline auto-detects GPU availability via `paddle.device.is_compiled_with_cuda`. To switch deployment modes, change your Helm values or Docker Compose configuration to include or exclude GPU resource reservations. No application code changes are needed.

**Q: How many CPU workers equal one GPU?**

A: With ONNX Runtime optimization: approximately 3-4 CPU workers (4 vCPU each) match the throughput of 1 NVIDIA T4 GPU. Without optimization (native PaddlePaddle CPU): approximately 10-15 CPU workers for equivalent throughput.

**Q: Should I use spot/preemptible instances?**

A: Yes, especially for CPU workers. The pipeline has page-level crash resume via the `ocr_temp/` directory, so a spot termination only loses the in-progress page (a few seconds of work). All completed pages are preserved. Set KEDA cooldown to 0 for spot-friendly scaling that avoids paying for idle instances.

**Q: Does the OCR accuracy differ between CPU and GPU?**

A: No. CPU and GPU run the same PaddleOCR models with identical weights. The inference results are mathematically equivalent (within floating-point precision, which does not affect character recognition). There is no accuracy trade-off when choosing CPU.

**Q: What about the (NER, classification, extraction, handwriting detection)?**

A: All , regardless of deployment mode. They execute in the assembler stage, not the OCR worker stage. They add zero GPU load and are unaffected by the CPU vs GPU decision for OCR inference.

**Q: What is the minimum hardware for a CPU-only deployment?**

A: A single worker needs at minimum 4 CPU cores and 8 GiB RAM. For production use, 6+ cores and 10+ GiB RAM per worker is recommended. The PaddleOCR models consume approximately 1-2 GiB of RAM per worker process.

**Q: Can I mix GPU and CPU workers in the same cluster?**

A: Yes. The distributed pipeline (Celery + RabbitMQ) supports heterogeneous worker pools. GPU workers subscribe to the `ocr_gpu` queue and CPU workers subscribe to the `ocr_cpu` queue. Task routing sends documents to the appropriate queue based on configuration or quality-based routing rules.

---

## Recommendations by Use Case

| Use Case | Recommended Deployment | Why |
|----------|----------------------|-----|
| Law firm (1K-10K pages/day) | CPU-only (ONNX, 2-4 workers) | Low volume, cost-sensitive, simple ops |
| Government agency (10K-50K/day) | CPU-only (ONNX, 8-12 workers) | Moderate volume, air-gapped requirement |
| Litigation support (50K-200K/day) | GPU (1-2x T4) or hybrid | High volume, deadline-driven |
| Insurance claims (variable burst) | Hybrid + autoscaling | Unpredictable volume, burst capacity needed |
| Medical records (5K-20K/day) | CPU-only (ONNX, 3-6 workers) | Low volume, privacy constraints |
| Enterprise archive (500K+/day) | GPU (4-8x T4/A10G) | Maximum throughput required |
| Development/testing | CPU-only (no optimization) | Convenience, no GPU needed |

---

Last Updated: 2026-05-20 | Pipeline Version: 4.1.0
