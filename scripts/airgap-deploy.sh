#!/usr/bin/env bash
# Air-gapped deployment script for OCR_LOCAL distributed pipeline.
#
# Loads pre-bundled Docker images and starts the coordinator stack.
# Run this in the directory created by airgap-bundle.sh.
#
# Usage:
#   tar xzf ocr_local_airgap_bundle_<date>.tar.gz
#   cd ocr_local_airgap_bundle_<date>
#   cp .env.template .env
#   # Edit .env with your secrets
#   ./airgap-deploy.sh
#
# Prerequisites:
#   - Docker Engine installed
#   - docker compose (v2) available
#   - NFS mount at /shared (or configure NFS_ROOT in .env)

set -euo pipefail

echo "=== OCR_LOCAL Air-Gapped Deployment ==="
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "ERROR: docker compose (v2) is not available."
    exit 1
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.template to .env and fill in your secrets first."
    exit 1
fi

# Check for images archive
if [ ! -f images.tar ]; then
    echo "ERROR: images.tar not found in current directory."
    echo "Make sure you're running this from the extracted bundle directory."
    exit 1
fi

echo "Step 1: Loading Docker images..."
docker load -i images.tar
echo "  Images loaded successfully."

echo ""
echo "Step 2: Starting coordinator stack..."
docker compose -f docker-compose.coordinator.yml up -d --no-build --pull never

echo ""
echo "Step 3: Waiting for services to be healthy..."
MAX_WAIT=120
INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    READY=true
    for service in postgres rabbitmq redis django celery-coordinator celery-beat flower; do
        container=$(docker compose -f docker-compose.coordinator.yml ps -q "$service")
        if [ -z "$container" ] || [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then
            READY=false
        fi
    done
    docker compose -f docker-compose.coordinator.yml exec -T postgres pg_isready -U ocr -d ocr_coordinator >/dev/null 2>&1 || READY=false
    docker compose -f docker-compose.coordinator.yml exec -T rabbitmq rabbitmq-diagnostics -q ping >/dev/null 2>&1 || READY=false
    docker compose -f docker-compose.coordinator.yml exec -T redis sh -c 'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli -e ping | grep -qx PONG' >/dev/null 2>&1 || READY=false
    docker compose -f docker-compose.coordinator.yml exec -T django python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3)" >/dev/null 2>&1 || READY=false
    docker compose -f docker-compose.coordinator.yml exec -T celery-coordinator sh -c 'celery -A coordinator inspect ping --destination "coordinator@$HOSTNAME" --timeout 3' >/dev/null 2>&1 || READY=false
    if $READY; then
        echo "  Infrastructure and application startup probes passed after ${ELAPSED}s."
        echo "  Real fixture OCR and operator production qualification remain separate gates."
        break
    fi
    echo "  Waiting for application startup... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "  ERROR: Services did not become healthy within ${MAX_WAIT}s." >&2
    echo "  Check logs with: docker compose -f docker-compose.coordinator.yml logs"
    exit 1
fi

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Services:"
echo "  Django Admin:  http://localhost:8000/admin/"
echo "  Flower:        http://localhost:5555/"
echo "  RabbitMQ Mgmt: http://localhost:15672/"
echo ""
echo "To start GPU workers on this node:"
echo "  docker compose -f docker-compose.worker.yml up -d --no-build --pull never"
echo ""
echo "To check status:"
echo "  docker compose -f docker-compose.coordinator.yml ps"
