#!/usr/bin/env bash
# Bind a local build/compose command to the current source, including dirty edits.
# Usage: bash scripts/with_runtime_source.sh root-cpu docker build -f Dockerfile ...
# or: bash scripts/with_runtime_source.sh coordinator docker compose -f coordinator/docker-compose.coordinator.yml build
# or: bash scripts/with_runtime_source.sh frontend docker build -f Dockerfile.frontend ...
set -euo pipefail
profile="${1:?runtime profile required: root-cpu, coordinator, worker, or frontend}"
shift
[[ $# -gt 0 ]] || { echo 'Build/compose command required' >&2; exit 2; }
repo_root="$(git rev-parse --show-toplevel)"
export SOURCE_REVISION="$(git -C "$repo_root" rev-parse HEAD)"
export SOURCE_TREE_SHA256="$(python3 "$repo_root/scripts/runtime_source_fingerprint.py" --repo "$repo_root" --profile "$profile")"
if [[ "${1:-}" == docker && "${2:-}" == build ]]; then
    shift 2
    exec docker build --build-arg SOURCE_REVISION --build-arg SOURCE_TREE_SHA256 "$@"
fi
exec "$@"
