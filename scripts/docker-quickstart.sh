#!/usr/bin/env bash
#
# EDCOCR Docker Quickstart
#
# One-shot Docker installation. Detects GPU automatically and runs a smoke test
# against the sample fixture. Exits 0 only if the end-to-end test produces output.
#
# Usage:
#   ./scripts/docker-quickstart.sh
#   ./scripts/docker-quickstart.sh --cpu-only

set -euo pipefail

CPU_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --cpu-only) CPU_ONLY=1 ;;
    --help|-h)
      cat <<EOF
EDCOCR Docker Quickstart

Usage: $0 [--cpu-only]

This script will:
  1. Detect GPU availability (or use --cpu-only if specified)
  2. Build the Docker stack
  3. Start all services
  4. Wait for services to be healthy
  5. Copy a sample PDF and run a smoke test
  6. Report success or failure with logs

EOF
      exit 0
      ;;
  esac
done

cd "$(dirname "$0")/.."

echo ""
echo "EDCOCR Docker Quickstart"
echo "=========================="
echo ""

# 1. Detect GPU
if [ "${CPU_ONLY}" = "0" ]; then
  if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[OK]    GPU detected"
    OVERLAY=""
  else
    echo "[WARN]  No GPU detected, using CPU-only stack"
    CPU_ONLY=1
  fi
fi

if [ "${CPU_ONLY}" = "1" ]; then
  OVERLAY="-f docker-compose.yml -f docker-compose.cpu-only.yml"
else
  OVERLAY=""
fi

# 2. Build
echo "[INFO]  Building Docker images..."
docker compose ${OVERLAY} build

# 3. Start
echo "[INFO]  Starting services..."
docker compose ${OVERLAY} up -d

# 4. Wait for healthy
echo "[INFO]  Waiting for services to become healthy..."
for i in {1..30}; do
  STATUS=$(docker compose ${OVERLAY} ps --format json 2>/dev/null || true)
  if echo "$STATUS" | grep -q "healthy"; then
    echo "[OK]    Services healthy"
    break
  fi
  if [ "$i" = "30" ]; then
    echo "[ERROR] Services did not become healthy in 90 seconds"
    docker compose ${OVERLAY} ps
    docker compose ${OVERLAY} logs --tail=50
    exit 1
  fi
  sleep 3
done

# 5. Smoke test
echo "[INFO]  Running smoke test..."
mkdir -p ocr_source ocr_output

if [ -f "tests/fixtures/sample.pdf" ]; then
  cp tests/fixtures/sample.pdf ocr_source/quickstart_sample.pdf
  echo "[OK]    Sample PDF staged"
else
  echo "[WARN]  tests/fixtures/sample.pdf not present"
fi

# Wait for the file watcher to pick it up
echo "[INFO]  Waiting for pipeline to process sample (up to 60s)..."
for i in {1..20}; do
  if ls ocr_output/EXPORT/PDF/quickstart_sample*.pdf &> /dev/null 2>&1; then
    echo "[OK]    Sample processed"
    break
  fi
  if [ "$i" = "20" ]; then
    echo "[WARN]  Sample did not process in 60 seconds — check logs:"
    echo "         docker compose ${OVERLAY} logs ocr_gpu_processor"
  fi
  sleep 3
done

# 6. Health endpoint
echo "[INFO]  Checking health endpoint..."
if curl -s -f http://localhost:8000/api/v1/health 2>&1; then
  echo ""
  echo "[OK]    Health endpoint responsive"
else
  echo "[WARN]  Health endpoint not responsive (services may still be starting)"
fi

# Summary
echo ""
echo "=========================="
echo "Quickstart complete"
echo "=========================="
echo ""
echo "Services:"
docker compose ${OVERLAY} ps
echo ""
echo "Next steps:"
echo "  - Drop more PDFs in:  ocr_source/"
echo "  - Watch results in:    ocr_output/EXPORT/PDF/"
echo "  - Tail logs:           docker compose ${OVERLAY} logs -f"
echo "  - API health:          curl http://localhost:8000/api/v1/health"
echo "  - Coordinator admin:   http://localhost:8001/admin/"
echo "  - Stop everything:     docker compose ${OVERLAY} down"
echo ""
