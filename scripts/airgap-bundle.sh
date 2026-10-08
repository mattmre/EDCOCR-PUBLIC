#!/usr/bin/env bash
# Air-gapped deployment bundle script for OCR_LOCAL distributed pipeline.
#
# Creates a tarball containing all Docker images needed to deploy the
# distributed OCR pipeline on a network with no internet access.
#
# Usage:
#   # Build images first:
#   export SOURCE_REVISION="$(git rev-parse HEAD)"
#   export SOURCE_TREE_SHA256="$(python3 scripts/runtime_source_fingerprint.py --profile coordinator)"
#   cd coordinator && docker compose -f docker-compose.coordinator.yml build
#   bash scripts/with_runtime_source.sh worker docker build -f coordinator/Dockerfile.worker.gpu -t ocr-worker:latest .
#
#   # Create bundle:
#   ./scripts/airgap-bundle.sh
#
# Output:
#   ocr_local_airgap_bundle_<date>.tar.gz
#
# Transfer this file to the air-gapped environment and use airgap-deploy.sh

set -euo pipefail

: "${SOURCE_REVISION:?Export SOURCE_REVISION from git rev-parse HEAD before building and bundling}"
: "${SOURCE_TREE_SHA256:?Export SOURCE_TREE_SHA256 with scripts/runtime_source_fingerprint.py --profile coordinator before building and bundling}"
: "${WORKER_SOURCE_TREE_SHA256:?Export WORKER_SOURCE_TREE_SHA256 with scripts/runtime_source_fingerprint.py --profile worker before building and bundling}"
if [[ ! "${SOURCE_REVISION}" =~ ^[[:xdigit:]]{40,64}$ ]]; then
    echo "ERROR: SOURCE_REVISION must be the full hexadecimal git commit ID." >&2
    exit 1
fi
if [[ ! "${SOURCE_TREE_SHA256}" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "ERROR: SOURCE_TREE_SHA256 must be the 64-character coordinator profile fingerprint." >&2
    exit 1
fi
if [[ ! "${WORKER_SOURCE_TREE_SHA256}" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "ERROR: WORKER_SOURCE_TREE_SHA256 must be the 64-character worker profile fingerprint." >&2
    exit 1
fi

BUNDLE_DATE=$(date +%Y%m%dT%H%M%SZ)
BUNDLE_NAME="ocr_local_airgap_bundle_${BUNDLE_DATE}"
BUNDLE_DIR="${BUNDLE_NAME}"
BUNDLE_TAR="${BUNDLE_NAME}.tar.gz"

if [[ -e "$BUNDLE_DIR" || -e "$BUNDLE_TAR" ]]; then
    echo "ERROR: refusing to replace an existing bundle" >&2
    exit 1
fi

echo "=== OCR_LOCAL Air-Gapped Bundle Builder ==="
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
        echo "  MISSING: $img — build with 'docker compose build' first" >&2
        exit 1
    fi
done

echo ""
echo "Step 3: Creating bundle directory..."
mkdir "$BUNDLE_DIR"

# Bind exact images before writing a large archive. Missing/mismatched
# coordinator labels cannot become a seemingly qualified offline bundle.
docker image inspect "${INFRA_IMAGES[@]}" "${APP_IMAGES[@]}" > "$BUNDLE_DIR/image-inspect.json"
python3 - "$BUNDLE_DIR/image-inspect.json" <<'PYCODE'
import json, os, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
for row in rows:
    tags = row.get("RepoTags") or []
    if any(tag.startswith("coordinator-") for tag in tags):
        labels = row.get("Config", {}).get("Labels") or {}
        if (labels.get("org.opencontainers.image.revision") != os.environ["SOURCE_REVISION"]
                or labels.get("io.ocr_local.runtime-source-sha256") != os.environ["SOURCE_TREE_SHA256"]
                or labels.get("io.ocr_local.runtime-profile") != "coordinator"):
            raise SystemExit("ERROR: bundled coordinator source identity mismatch")
    if "ocr-worker:latest" in tags:
        labels = row.get("Config", {}).get("Labels") or {}
        if (labels.get("org.opencontainers.image.revision") != os.environ["SOURCE_REVISION"]
                or labels.get("io.ocr_local.runtime-source-sha256") != os.environ["WORKER_SOURCE_TREE_SHA256"]
                or labels.get("io.ocr_local.runtime-profile") != "worker"):
            raise SystemExit("ERROR: bundled worker source identity mismatch")
manifest = [{"id": row["Id"], "tags": row.get("RepoTags") or [],
             "platform": row.get("Os", "") + "/" + row.get("Architecture", ""),
             "labels": row.get("Config", {}).get("Labels") or {}} for row in rows]
Path(sys.argv[1]).with_name("image-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
Path(sys.argv[1]).unlink()
PYCODE

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
# Preserve placeholders without serializing the build host's credentials.
# Explicit image references and no build blocks make the offline files usable
# from any project/directory name with --no-build --pull never.
for role in coordinator worker; do
    docker compose -f "coordinator/docker-compose.${role}.yml" config --no-interpolate --no-path-resolution --format json > "$BUNDLE_DIR/docker-compose.${role}.yml"
done
python3 - "$BUNDLE_DIR" <<'PYCODE'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for role in ("coordinator", "worker"):
    path = root / ("docker-compose." + role + ".yml")
    data = json.loads(path.read_text())
    services = data["services"]
    for name in list(services):
        if name == "edc-translation":
            del services[name]  # optional external app is not in this bundle
            continue
        service = services[name]
        if "build" in service:
            del service["build"]
            service["image"] = "ocr-worker:latest" if role == "worker" else "coordinator-" + name + ":latest"
    path.write_text(json.dumps(data, indent=2) + "\n")
PYCODE
mkdir -p "${BUNDLE_DIR}/rabbitmq"
cp coordinator/rabbitmq/definitions.json "${BUNDLE_DIR}/rabbitmq/definitions.json"
cp scripts/airgap-deploy.sh "${BUNDLE_DIR}/"
chmod +x "${BUNDLE_DIR}/airgap-deploy.sh"

# Create a .env template
cat > "${BUNDLE_DIR}/.env.template" << 'ENVEOF'
# Required secrets for OCR_LOCAL distributed pipeline
# Copy this file to .env and fill in values before deploying.
POSTGRES_PASSWORD=changeme
RABBITMQ_PASSWORD=changeme
REDIS_PASSWORD=changeme
DJANGO_SECRET_KEY=changeme-use-a-long-random-string
FLOWER_PASSWORD=changeme
NFS_ROOT=/shared
CELERY_TASK_SIGNING_KEY=
PII_ENCRYPTION_KEY=
WEBHOOK_SECRET_KEY=
# Worker nodes also require operator-supplied coordinator connection URLs:
CELERY_BROKER_URL=
CELERY_RESULT_BACKEND=
DATABASE_URL=
ENVEOF

# The offline deploy compose still parses build args even though it uses the
# prebuilt images. Carry the source identity through .env so target-side
# compose commands work without source checkout or helper script.
cat >> "${BUNDLE_DIR}/.env.template" << ENVEOF
SOURCE_REVISION=${SOURCE_REVISION}
SOURCE_TREE_SHA256=${SOURCE_TREE_SHA256}
WORKER_SOURCE_TREE_SHA256=${WORKER_SOURCE_TREE_SHA256}
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
