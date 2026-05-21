# Known Issues

## Active Issues

### KI-001: Docker stdout buffering
- **Symptom**: Log output appears delayed or batched when running `docker logs -f`
- **Cause**: Python stdout buffering in Docker
- **Workaround**: `python3 -u` flag (already used in docker-compose.yml), PrintLogger class with `flush=True`
- **Status**: Mitigated

### KI-002: PaddleOCR model download race condition
- **Symptom**: Multiple workers trying to download same model simultaneously on first run
- **Cause**: `model_load_lock` prevents concurrent model loading, but download itself may still race
- **Workaround**: Pre-download PaddleOCR models in Dockerfile via `download_models.py`. FastText `lid.176.bin` is handled separately by the Dockerfile build and does not rely on runtime download.
- **Status**: Mitigated (PaddleOCR models only)

### KI-003: Large PDF memory spikes
- **Symptom**: Processing 50,000+ page PDFs can spike RAM usage during assembly
- **Cause**: Assembler loads all page chunks for merge
- **Workaround**: Keep IMAGE_QUEUE_SIZE low, monitor with `docker stats`
- **Status**: Open

### KI-004: Tesseract fallback quality
- **Symptom**: When PaddleOCR fails and Tesseract takes over, text quality drops significantly
- **Cause**: Tesseract doesn't use GPU acceleration, lower accuracy on complex layouts
- **Workaround**: None — this is by design as a last-resort fallback
- **Status**: By design

### KI-005: Resume may re-process pages unnecessarily
- **Symptom**: After crash, some pages that completed OCR but weren't assembled may be re-OCR'd
- **Cause**: Resume checks temp dir for .pdf files but doesn't verify content integrity
- **Workaround**: Accept minor re-processing overhead
- **Status**: Open (low priority)
