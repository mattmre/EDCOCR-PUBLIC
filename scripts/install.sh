#!/usr/bin/env bash
#
# EDCOCR Installer (Linux/macOS)
#
# Usage:
#   ./scripts/install.sh             # interactive install
#   ./scripts/install.sh --docker    # Docker install only
#   ./scripts/install.sh --bare      # bare-metal Python only
#   ./scripts/install.sh --help
#
# Requires: bash 4+, curl, git
# Adds optional: Docker, Python 3.11, Poppler, Tesseract

set -euo pipefail

EDCOCR_VERSION="4.1.0"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
fatal() { err "$*"; exit 1; }

print_banner() {
  cat <<'BANNER'

   _____ ____   _____ ____  _____ ____
  | ____|  _ \ / ____/ __ \/ ____|  _ \
  | |__ | | | | |   | |  | | |    | |_) |
  |  __|| | | | |   | |  | | |    |  _ <
  | |___| |_| | |___| |__| | |____| |_) |
  |_____|____/ \_____\____/ \_____|____/

  Forensic-Grade OCR Platform
  Version 4.1.0

BANNER
}

print_help() {
  cat <<EOF
EDCOCR Installer

Usage:
  $0 [OPTIONS]

Options:
  --docker     Install with Docker (recommended)
  --bare       Bare-metal Python install
  --cpu-only   CPU-only install (no GPU required)
  --help, -h   Show this help message

Examples:
  $0                    # Interactive install
  $0 --docker           # Docker install with GPU
  $0 --docker --cpu-only # Docker install without GPU
  $0 --bare             # Python install on host

EOF
}

check_os() {
  case "$(uname -s)" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="macos" ;;
    *)       fatal "Unsupported OS: $(uname -s). Linux or macOS required." ;;
  esac
  ok "Detected OS: ${OS}"
}

check_docker() {
  if ! command -v docker &> /dev/null; then
    err "Docker is not installed."
    log "Install Docker from: https://docs.docker.com/engine/install/"
    return 1
  fi

  if ! docker compose version &> /dev/null; then
    err "Docker Compose v2 not available."
    log "Update Docker to a version that includes 'docker compose'."
    return 1
  fi

  ok "Docker $(docker --version | awk '{print $3}' | tr -d ',') available"
  return 0
}

check_gpu() {
  if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
      ok "NVIDIA GPU detected"
      return 0
    fi
  fi
  warn "No NVIDIA GPU detected (CPU-only mode will be used)"
  return 1
}

check_python() {
  if ! command -v python3 &> /dev/null; then
    err "Python 3 not installed."
    return 1
  fi

  PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
  PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)

  if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    err "Python 3.10+ required (found: ${PYTHON_VER})"
    return 1
  fi

  ok "Python ${PYTHON_VER} available"
  return 0
}

install_docker() {
  log "Building Docker images (this may take 10-15 minutes on first run)..."

  cd "${REPO_ROOT}"

  if [ "${CPU_ONLY:-0}" = "1" ]; then
    log "Using CPU-only compose overlay"
    docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml build
    docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d
  else
    docker compose build
    docker compose up -d
  fi

  log "Waiting for services to become healthy (up to 90 seconds)..."
  for i in {1..18}; do
    if docker compose ps 2>/dev/null | grep -q "(healthy)"; then
      break
    fi
    sleep 5
  done

  echo ""
  docker compose ps

  ok "Docker install complete"
  log "API:        http://localhost:8000"
  log "Coordinator: http://localhost:8001"
  log "Logs:       docker compose logs -f"
  log "Stop:       docker compose down"
}

install_bare() {
  log "Installing Python dependencies..."

  cd "${REPO_ROOT}"

  if [ ! -d ".venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv .venv
  fi

  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

  if [ "${CPU_ONLY:-0}" = "1" ]; then
    log "Pre-downloading models (CPU-only)..."
    python download_models.py --cpu-only
  else
    log "Pre-downloading models..."
    python download_models.py
  fi

  ok "Bare-metal install complete"
  log "Activate venv: source .venv/bin/activate"
  log "Run pipeline:  python ocr_gpu_async.py"
  log "Run API:       uvicorn api.main:app --host 0.0.0.0 --port 8000"
}

run_smoke_test() {
  log "Running smoke test..."

  cd "${REPO_ROOT}"

  if [ ! -f "tests/fixtures/sample.pdf" ]; then
    warn "tests/fixtures/sample.pdf not present, skipping smoke test"
    return 0
  fi

  if [ "${INSTALL_MODE}" = "docker" ]; then
    docker exec ocr_gpu_processor python /app/scripts/smoke_pipeline.py 2>&1 || warn "Smoke test had issues — check docker compose logs"
  else
    source .venv/bin/activate
    python scripts/smoke_pipeline.py || warn "Smoke test had issues — check pipeline output"
  fi
}

main() {
  print_banner

  INSTALL_MODE=""
  CPU_ONLY=0

  while [ $# -gt 0 ]; do
    case "$1" in
      --docker)    INSTALL_MODE="docker"; shift ;;
      --bare)      INSTALL_MODE="bare"; shift ;;
      --cpu-only)  CPU_ONLY=1; shift ;;
      --help|-h)   print_help; exit 0 ;;
      *)           err "Unknown option: $1"; print_help; exit 1 ;;
    esac
  done

  check_os

  if [ -z "${INSTALL_MODE}" ]; then
    echo ""
    log "Choose install mode:"
    echo "  1) Docker (recommended)"
    echo "  2) Bare-metal Python"
    echo ""
    read -p "Selection [1]: " choice
    choice="${choice:-1}"
    case "$choice" in
      1) INSTALL_MODE="docker" ;;
      2) INSTALL_MODE="bare" ;;
      *) fatal "Invalid selection" ;;
    esac
  fi

  if [ "${CPU_ONLY}" = "0" ]; then
    check_gpu || CPU_ONLY=1
  fi

  export CPU_ONLY INSTALL_MODE

  case "${INSTALL_MODE}" in
    docker)
      check_docker || fatal "Docker prerequisites not met"
      install_docker
      ;;
    bare)
      check_python || fatal "Python prerequisites not met"
      install_bare
      ;;
  esac

  echo ""
  ok "EDCOCR ${EDCOCR_VERSION} installed"
  echo ""
  log "Next steps:"
  echo "  - Drop documents in: ${REPO_ROOT}/ocr_source/"
  echo "  - Results land in:    ${REPO_ROOT}/ocr_output/EXPORT/"
  echo "  - Read INSTALL.md for verification steps"
  echo "  - Read docs/02-QUICKSTART-5-MINUTE-SUCCESS.md for a guided walkthrough"
  echo ""
}

main "$@"
