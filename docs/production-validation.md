# Production Validation Checklist

Systematic test matrix for validating the OCR pipeline before production deployment.

## Prerequisites

```bash
# Build the container
docker-compose up -d --build

# Verify container is running and healthy
docker ps --filter name=ocr_gpu_processor
docker logs --tail 50 ocr_gpu_processor
```

## 1. Basic Pipeline (Default Flags)

| # | Test Case | Input | Expected Output | Pass? |
|---|-----------|-------|-----------------|-------|
| 1.1 | Single-page PDF | 1-page text PDF | `EXPORT/PDF/*.pdf` + `EXPORT/TEXT/*.txt` | |
| 1.2 | Multi-page PDF | 10+ page PDF | All pages in output, correct page count | |
| 1.3 | Scanned image PDF | Image-only PDF (no text layer) | OCR text extracted via Paddle/Tesseract | |
| 1.4 | Single TIFF image | `.tif` file | Converted to searchable PDF + text | |
| 1.5 | Single JPEG image | `.jpg` file | Converted to searchable PDF + text | |
| 1.6 | PNG image | `.png` file | Converted to searchable PDF + text | |
| 1.7 | Multi-page TIFF | Multi-frame `.tif` | Each frame as separate page in output PDF | |
| 1.8 | Empty source dir | No files in `ocr_source/` | Pipeline completes with no output | |
| 1.9 | Mixed batch | PDFs + images together | All processed, correct file mapping | |

## 2. Language Detection

| # | Test Case | Input | Expected | Pass? |
|---|-----------|-------|----------|-------|
| 2.1 | English document | English-only PDF | `[en]` in scheduler log | |
| 2.2 | Non-English document | French/German/Spanish PDF | Correct language code in log | |
| 2.3 | Mixed-language doc | English + Chinese text | Language detected, model swapped | |
| 2.4 | CJK document | Chinese/Japanese/Korean text | Correct CJK model loaded | |

## 3. OCR Fallback Chain

| # | Test Case | Input | Expected | Pass? |
|---|-----------|-------|----------|-------|
| 3.1 | Paddle succeeds | Clean typed document | Status: `OK` in assembly | |
| 3.2 | Paddle fails → Tesseract | Degraded/unusual scan | Status: `Tesseract` fallback logged | |
| 3.3 | Both fail → ImageOnly | Severely corrupted image | Status: `ImageOnly`, image preserved in PDF | |
| 3.4 | Corrupt PDF page | PDF with damaged page stream | Failure logged, other pages processed | |

## 4. Document Intelligence (`--enable-docintel`)

| # | Test Case | Command | Expected | Pass? |
|---|-----------|---------|----------|-------|
| 4.1 | Layout analysis | `--enable-docintel` | `EXPORT/STRUCTURE/*.json` with layout_regions | |
| 4.2 | Table extraction | `--enable-docintel` | Tables detected in structure JSON | |
| 4.3 | Layout-only mode | `--enable-docintel --docintel-mode layout_only` | Layout regions only, no tables | |
| 4.4 | Tables-only mode | `--enable-docintel --docintel-mode tables_only` | Tables only, no layout | |
| 4.5 | Table export | `--enable-docintel --export-tables` | HTML/CSV table files generated | |
| 4.6 | DocIntel disabled (default) | No flags | No `STRUCTURE/` directory created | |

## 5. Chain of Custody

| # | Test Case | Command | Expected | Pass? |
|---|-----------|---------|----------|-------|
| 5.1 | Custody enabled (default) | Default run | `EXPORT/CUSTODY/*.custody.jsonl` created | |
| 5.2 | Custody disabled | `--no-custody` | No `CUSTODY/` directory | |
| 5.3 | Chain integrity | Default run | `verify_custody_file` returns True | |
| 5.4 | Full lifecycle events | Default run | Events: file_ingested → ocr_* → assembly_complete → compression_complete | |
| 5.5 | Failure events | Corrupt input | `processing_failed` event in chain | |

## 6. Compression (Ghostscript)

| # | Test Case | Input | Expected | Pass? |
|---|-----------|-------|----------|-------|
| 6.1 | Standard compression | Normal PDF | Output PDF smaller than uncompressed | |
| 6.2 | Already compressed | Pre-optimized PDF | No corruption, file integrity maintained | |
| 6.3 | Large document | 100+ page PDF | Compression completes without OOM | |

## 7. Crash Resume

| # | Test Case | Steps | Expected | Pass? |
|---|-----------|-------|----------|-------|
| 7.1 | Interrupt and resume | Stop container mid-processing, restart | Resumes from last completed page | |
| 7.2 | Temp dir preserved | Check `ocr_temp/` after interrupt | Per-page PDFs present for completed pages | |
| 7.3 | Partial re-run | Re-run with same input | Only missing pages processed | |

## 8. REST API

| # | Test Case | Endpoint | Expected | Pass? |
|---|-----------|----------|----------|-------|
| 8.1 | Health check | `GET /api/v1/health` | 200 with version + uptime | |
| 8.2 | Submit file upload | `POST /api/v1/jobs` (multipart) | 201 with job_id | |
| 8.3 | Submit source path | `POST /api/v1/jobs` (source_path) | 201 with job_id | |
| 8.4 | Job status | `GET /api/v1/jobs/{id}` | 200 with progress | |
| 8.5 | Job list | `GET /api/v1/jobs` | 200 with pagination | |
| 8.6 | Download result | `GET /api/v1/jobs/{id}/result/download` | PDF file download | |
| 8.7 | Cancel job | `DELETE /api/v1/jobs/{id}` | 200 with cancelled status | |
| 8.8 | Retry failed job | `POST /api/v1/jobs/{id}/retry` | 201 with new job_id | |
| 8.9 | Auth required | No `X-API-Key` header | 401 Unauthorized | |
| 8.10 | Rate limiting | Exceed request limit | 429 Too Many Requests | |

## 9. Graceful Shutdown

| # | Test Case | Steps | Expected | Pass? |
|---|-----------|-------|----------|-------|
| 9.1 | SIGTERM | `docker stop ocr_gpu_processor` | Clean shutdown logged, no data loss | |
| 9.2 | SIGINT | Ctrl+C in attached mode | Clean shutdown logged | |
| 9.3 | Docker healthcheck | Wait 2+ minutes | Healthcheck passes (exit 0) | |

## 10. Throughput Metrics

Record these from the monitor thread logs during a representative batch:

| Metric | Value | Notes |
|--------|-------|-------|
| Pages per minute (PPM) | | From monitor log |
| Documents per hour | | From monitor log |
| Average page processing time | | Calculated |
| Peak GPU memory usage | | `nvidia-smi` during run |
| Peak RAM usage | | `docker stats` during run |
| Queue depths (peak) | | From monitor log |

## Verification Commands

```bash
# Check output structure
find ocr_output/EXPORT -type f | head -20

# Verify custody chain integrity
python3 -c "
from custody import verify_custody_file
import glob
for f in glob.glob('ocr_output/EXPORT/CUSTODY/*.custody.jsonl'):
    ok, msg = verify_custody_file(f)
    print(f'{f}: {msg}')
"

# Count processed files
find ocr_output/EXPORT/PDF -name '*.pdf' | wc -l
find ocr_output/EXPORT/TEXT -name '*.txt' | wc -l

# Check for failures
cat ocr_output/failures.csv

# API health check (if API server running)
curl -H "X-API-Key: $OCR_API_KEY" http://localhost:8000/api/v1/health
```
