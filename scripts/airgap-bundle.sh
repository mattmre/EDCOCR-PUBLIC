#!/usr/bin/env bash
# Air-gapped deployment bundle script for EDCOCR distributed pipeline.
#
# Creates a tarball containing all Docker images needed to deploy the
# distributed OCR pipeline on a network with no internet access.
#
# Usage:
#   # Build images first:
#   cd coordinator && docker compose -f docker-compose.coordinator.yml build
#   docker build -f coordinator/Dockerfile.worker -t ocr-worker:latest .
#
#   # Create bundle:
#   ./scripts/airgap-bundle.sh
#
# Output:
#   ocr_local_airgap_bundle_<date>.tar.gz
#
# Transfer this file to the air-gapped environment and use airgap-deploy.sh

set -euo pipefail

BUNDLE_DATE=$(date +%Y%m%d)
BUNDLE_NAME="ocr_local_airgap_bundle_${BUNDLE_DATE}"
BUNDLE_DIR="${BUNDLE_NAME}"
BUNDLE_TAR="${BUNDLE_NAME}.tar.gz"

echo "=== EDCOCR Air-Gapped Bundle Builder ==="
echo ""

# Infrastructure images
INFRA_IMAGES=(
    "postgres:16-alpine"
    "rabbitmq:3.13-management-alpine"
    "redis:7-alpine"
)

# Application images (must be built first)
APP_IMAGES=(
    "coordinator-django:latest"
    "coordinator-celery-coordinator:latest"
    "coordinator-celery-beat:latest"
    "coordinator-flower:latest"
    "ocr-worker:latest"
)

echo "Step 1: Pulling infrastructure images..."
for img in "${INFRA_IMAGES[@]}"; do
    echo "  Pulling $img"
    if ! docker pull "$img"; then
        # Check if the image exists locally despite pull failure
        if docker image inspect "$img" >/dev/null 2>&1; then
            echo "  WARNING: Pull failed but $img exists locally — using cached version"
        else
            echo "  ERROR: Could not pull $img and no local copy exists"
            exit 1
        fi
    fi
done

echo ""
echo "Step 2: Checking application images..."
for img in "${APP_IMAGES[@]}"; do
    if docker image inspect "$img" >/dev/null 2>&1; then
        echo "  Found: $img"
    else
        echo "  MISSING: $img — build with 'docker compose build' first"
    fi
done

echo ""
echo "Step 3: Creating bundle directory..."
mkdir -p "$BUNDLE_DIR"

echo "Step 4: Saving Docker images..."
ALL_IMAGES=("${INFRA_IMAGES[@]}" "${APP_IMAGES[@]}")
EXISTING_IMAGES=()
for img in "${ALL_IMAGES[@]}"; do
    if docker image inspect "$img" >/dev/null 2>&1; then
        EXISTING_IMAGES+=("$img")
    fi
done

if [ ${#EXISTING_IMAGES[@]} -eq 0 ]; then
    echo "ERROR: No images found to bundle."
    exit 1
fi

echo "  Saving ${#EXISTING_IMAGES[@]} images to ${BUNDLE_DIR}/images.tar..."
docker save "${EXISTING_IMAGES[@]}" -o "${BUNDLE_DIR}/images.tar"

echo "Step 5: Copying deployment files..."
cp coordinator/docker-compose.coordinator.yml "${BUNDLE_DIR}/"
cp coordinator/docker-compose.worker.yml "${BUNDLE_DIR}/"
cp scripts/airgap-deploy.sh "${BUNDLE_DIR}/"
chmod +x "${BUNDLE_DIR}/airgap-deploy.sh"

# Create a .env template
cat > "${BUNDLE_DIR}/.env.template" << 'ENVEOF'
# Required secrets for EDCOCR distributed pipeline
# Copy this file to .env and fill in values before deploying.
POSTGRES_PASSWORD=changeme
RABBITMQ_PASSWORD=changeme
REDIS_PASSWORD=changeme
DJANGO_SECRET_KEY=changeme-use-a-long-random-string
FLOWER_PASSWORD=changeme
NFS_ROOT=/shared
ENVEOF

echo "Step 6: Creating tarball..."
tar -czf "$BUNDLE_TAR" "$BUNDLE_DIR"
rm -rf "$BUNDLE_DIR"

BUNDLE_SIZE=$(du -h "$BUNDLE_TAR" | cut -f1)
echo ""
echo "=== Bundle created: ${BUNDLE_TAR} (${BUNDLE_SIZE}) ==="
echo ""
echo "Transfer this file to the air-gapped environment and run:"
echo "  tar xzf ${BUNDLE_TAR}"
echo "  cd ${BUNDLE_NAME}"
echo "  ./airgap-deploy.sh"
