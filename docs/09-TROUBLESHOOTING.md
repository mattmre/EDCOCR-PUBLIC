# 09: Troubleshooting

## Quick Diagnostics

Before diving into specific issues, run through this checklist:

```bash
curl http://localhost:8000/api/v1/health
docker compose ps
docker compose logs --tail=50 ocr-pipeline
tail -100 ocr_output/logs/pipeline.log
cat ocr_output/failures.csv
```

> [!TIP]
> The monitor thread writes a heartbeat to `HEALTHCHECK_FILE` every 30 seconds.

---

## Common Issues

### Pipeline Issues

| Symptom | Cause | Fix |
|---|---|---|
| Pipeline hangs on startup | GPU not detected | Check `nvidia-smi` and install NVIDIA Container Toolkit |
| Low OCR confidence | Poor scan quality | Enable DPI escalation and lower thresholds |
| Blank output PDFs | Ghostscript missing | Install Ghostscript and verify PATH |
| Language detection wrong | FastText model missing | Verify `lid.176.bin` exists at `FASTTEXT_MODEL_PATH` |
| OOM | Too many workers | Reduce `NUM_WORKERS` |
| Pages missing from output | Crash during processing | Inspect `ocr_temp/` and re-run |

### API Issues

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Missing or invalid API key | Set `OCR_API_KEY` and send `X-API-Key` |
| `403 Forbidden` | RBAC or allowlist failure | Check `API_ALLOWED_IPS` and role |
| `429 Too Many Requests` | Rate limit exceeded | Back off and check rate-limit env vars |
| `503 server_misconfigured` | API key required but not configured | Set `OCR_API_KEY` |

### Distributed Coordinator Issues

| Symptom | Cause | Fix |
|---|---|---|
| Workers not picking up tasks | Broker misconfigured | Verify `CELERY_BROKER_URL` |
| Jobs stuck in processing | Worker crashed | Enable stale job cleanup |
| PostgreSQL refused | Database not started | Check coordinator DB settings |

---

## Storage and Infrastructure Issues

### S3 / MinIO Connection Issues

| Symptom | Cause | Fix |
|---|---|---|
| `EndpointConnectionError` | `S3_ENDPOINT_URL` wrong or unreachable | Verify endpoint URL, TLS, and that the port is reachable from workers; test with `aws --endpoint-url=$S3_ENDPOINT_URL s3 ls` |
| `InvalidAccessKeyId` / `SignatureDoesNotMatch` | `S3_ACCESS_KEY` / `S3_SECRET_KEY` mismatch or placeholder (`minioadmin`) | Rotate credentials, redeploy with updated secret, avoid default MinIO creds |
| `NoSuchBucket` | `S3_BUCKET` not created, wrong region, or misspelled | Create the bucket: `aws --endpoint-url=... s3 mb s3://$S3_BUCKET`; check `S3_REGION` |
| Worker uploads succeed but URLs expire too quickly | `S3_PRESIGN_EXPIRY_SECONDS` too low | Raise to match worst-case job duration (default 3600s); confirm clock skew between worker and S3 |
| `SlowDown` / `503` from MinIO | Too many parallel uploads | Lower `NUM_COMPRESSORS` or split workload across tenants |
| Presigned URLs return 403 from worker | Worker clock drift or shared tenant key missing | Sync NTP across workers; confirm the coordinator and worker share `S3_PRESIGN_*` config |
| "Storage backend not configured" on job submit | `STORAGE_BACKEND` unset or invalid | Set `STORAGE_BACKEND=s3` (or `nfs`) and restart the API/coordinator |

Related: `coordinator/jobs/storage.py`, `docs/operations/production-cutover-runbook.md`, `scripts/migrate_nfs_to_s3.py`.

### KEDA Autoscaling Not Triggering

| Symptom | Cause | Fix |
|---|---|---|
| Worker count never scales up under load | `ScaledObject` not applied in the right namespace | `kubectl get scaledobject -n <ns>`; re-install Helm chart with `--set keda.enabled=true` |
| `queueLength` trigger reports 0 | KEDA cannot reach RabbitMQ management API | Verify `host`/`protocol` in the trigger metadata and that the management plugin is enabled (`rabbitmq_management`) |
| Scaling works but workers thrash between 0 and N | `cooldownPeriod` / `pollingInterval` too aggressive | Raise `cooldownPeriod` (default 300s) and confirm `minReplicaCount` matches steady-state load |
| KEDA logs show `unauthorized` | RabbitMQ user missing `monitoring` tag | `rabbitmqctl set_user_tags <user> monitoring`; restart KEDA operator |
| `ScaledObject` ready but HPA missing | CRD version mismatch with KEDA operator | Upgrade KEDA to >=2.13 and re-apply the chart |
| Queue length stuck after worker crashes | Unacked messages not returned to queue | Enable quorum queues (`rabbitmq.quorumQueues=true`) and restart consumers |

Related: `helm/ocr-local/templates/keda-*`, `helm/ocr-local/values.yaml` (`keda.*` block).

### LayoutLMv3 Model Loading Failures

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: model weights not found` | Model not pre-baked in worker image | Rebuild with `download_models.py` or mount the model cache volume; confirm `LAYOUTLM_MODEL_PATH` |
| `LAYOUTLM_ACTIVE_MODEL` not resolving | Registry not wired or alias missing | Check `layoutlm_model_registry.py` and that `semantic_extraction.py` + `coordinator/jobs/tasks_layoutlm.py` both read the alias |
| `CUDA out of memory` during LayoutLM inference | Model loaded alongside PaddleOCR on the same GPU | Reduce `NUM_WORKERS`, or run LayoutLM on a dedicated GPU via per-GPU queue affinity (`ENABLE_PER_GPU_QUEUES=true`) |
| Inference runs but entities are empty | Model/tokenizer version mismatch | Re-download via `download_models.py`; confirm `transformers` version matches what the checkpoint was exported with |
| Confidence calibration off | Calibration file missing | Re-run `layoutlm_calibration.py` and commit the new curve to the model registry |

Related: `layoutlm_model_registry.py`, `docs/11-ML-TRAINING-GUIDE.md`.

### Video Ingestion Failures

| Symptom | Cause | Fix |
|---|---|---|
| Video file rejected with "unsupported format" | Container outside the accepted set | Re-encode to mp4/mov/mkv/webm/avi; verify extension against `ocr_distributed/constants.py` |
| Only the first frame is OCR'd | `VIDEO_MAX_FRAMES=1` | Raise `VIDEO_MAX_FRAMES` and check `VIDEO_FRAME_SAMPLE_SECONDS` |
| Pipeline times out on long videos | Default `JOB_PROCESSING_TIMEOUT_MINUTES` too low | Raise the global default or pass a per-job `processing_timeout_minutes` override |
| No frames extracted (empty output) | ffmpeg / opencv not installed in image | Rebuild the worker image; confirm `cv2.VideoCapture` returns `isOpened == True` |
| Frame extraction OK but OCR finds nothing | Low-motion video, frames look similar | Lower `VIDEO_FRAME_SAMPLE_SECONDS` to sample more aggressively |

Related: `ocr_distributed/constants.py`, `ocr_gpu_async.py` (video branch).

### Redis Sentinel Failover

| Symptom | Cause | Fix |
|---|---|---|
| Clients connect but fail over never happens | `sentinel.enabled=false` in Helm values | Re-install with `--set redis.sentinel.enabled=true`; confirm 3+ sentinel replicas |
| Sentinel elects new master but clients stick to old one | Using a direct connection string instead of sentinel URL | Switch connection string to `redis+sentinel://sentinel-a:26379,sentinel-b:26379,sentinel-c:26379/mymaster/0` |
| Failover takes >60s | `down-after-milliseconds` too high | Lower to 5000ms for production workloads |
| Split-brain after network partition | Only 2 sentinel replicas | Always run an odd number >=3; deploy across failure domains |
| Coordinator Celery loses tasks during failover | Broker URL only points to primary | Use a sentinel-aware broker URL or `rabbitmq` as broker + Redis only as result backend |
| Sentinel logs "no good slave" | All replicas unreachable | Check `redis-cli -h <replica> ping`, network policies, and Helm `redis.replicaCount` |

Related: `docs/FAILOVER-RUNBOOK.md` (Section 4), `helm/ocr-local/templates/redis-sentinel-statefulset.yaml`.

### Docker Build Errors

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA version mismatch` during `pip install paddlepaddle-gpu` | Base image CUDA version does not match the Paddle wheel | Pin the `nvidia/cuda:*` base image in the Dockerfile to match the wheel; rebuild with `--no-cache` |
| `poppler-utils: command not found` at runtime | Base image lacks poppler | Ensure the Dockerfile installs `poppler-utils` (required by `pdf2image`) |
| Model preload step hangs or segfaults (exit 139) | PaddleOCR probes CUDA during model download inside the build | Pass `--cpu-only` to `download_models.py` or set `CPU_ONLY_BUILD=1`; escape hatch: `SKIP_MODEL_PRELOAD=1` |
| `fasttext` model download 404 | FastText upstream moved | Use the `ADD` instruction with a pinned URL or bundle `lid.176.bin` into the build context |
| `USER 1000` cannot write to `/app/ocr_output` | Volume owned by root on host | Create the directory as the host user (UID 1000) or add an `initContainer`/`chown` step |
| Build succeeds but runtime fails with "no space left on device" | Multi-stage layers bloated | Use the multi-stage Dockerfile (`Dockerfile.worker`) and confirm the final image drops build-only tooling |
| Font glitches on CJK documents | Noto CJK font download drift | Pin to the specific upstream commit hash in the `ADD` instruction |

Related: `Dockerfile`, `Dockerfile.worker.*`, `download_models.py`.

---

## Debug Flags

| Flag / Env Var | Effect |
|---|---|
| `LOG_LEVEL=DEBUG` | Verbose pipeline logging |
| `ENABLE_PROFILING=true` | Write per-stage timing |
| `PADDLE_DEBUG=1` | Enable PaddleOCR debug output |
| `API_DEBUG=true` | Enable FastAPI debug mode |

---

## Getting Help

1. Check `FAILOVER-RUNBOOK.md` and `API-REFERENCE.md`.
2. Review `failures.csv`.
3. Enable debug logging temporarily.
