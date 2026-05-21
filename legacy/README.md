# Legacy Scripts

These scripts are kept for reference only and are **not used in production**.

## OCRLOCAL.py (v1)
- Original Tesseract-only OCR script
- Hardcoded Windows paths (`C:\Program Files\Tesseract-OCR\`)
- No GPU acceleration, no PaddleOCR
- Superseded by `OCR_GPU.py` (v2) and `ocr_gpu_async.py` (v3/production)

## OCR_GPU_sync_legacy.py (v2)
- Legacy synchronous PaddleOCR pipeline preserved for compatibility only
- Moved out of repo root to avoid unsafe import-time side effects
- Invoked only via explicit opt-in: `python OCR_GPU.py --run-legacy-sync`

**Do not use these scripts for production workloads.**
