# EDCOCR Installer (Windows PowerShell)
#
# Usage:
#   .\scripts\install.ps1                     # interactive
#   .\scripts\install.ps1 -Mode Docker        # Docker install
#   .\scripts\install.ps1 -Mode Bare          # bare-metal Python
#   .\scripts\install.ps1 -CpuOnly            # force CPU-only
#   .\scripts\install.ps1 -Help

param(
    [ValidateSet("Docker", "Bare", "")]
    [string]$Mode = "",
    [switch]$CpuOnly,
    [switch]$Help
)

$EdcocrVersion = "4.1.0"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

function Write-Banner {
    Write-Host @"

   _____ ____   _____ ____  _____ ____
  | ____|  _ \ / ____/ __ \/ ____|  _ \
  | |__ | | | | |   | |  | | |    | |_) |
  |  __|| | | | |   | |  | | |    |  _ <
  | |___| |_| | |___| |__| | |____| |_) |
  |_____|____/ \_____\____/ \_____|____/

  Forensic-Grade OCR Platform
  Version $EdcocrVersion

"@ -ForegroundColor Cyan
}

function Write-Info  { Write-Host "[INFO]  $args" -ForegroundColor Blue }
function Write-Ok    { Write-Host "[OK]    $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Err   { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Show-Help {
    Write-Host @"
EDCOCR Installer (Windows)

Usage:
  .\scripts\install.ps1 [-Mode <Docker|Bare>] [-CpuOnly] [-Help]

Parameters:
  -Mode      Install mode: Docker (recommended) or Bare (Python on host)
  -CpuOnly   Skip GPU detection and install CPU-only stack
  -Help      Show this help

Examples:
  .\scripts\install.ps1                       # interactive
  .\scripts\install.ps1 -Mode Docker          # Docker with GPU
  .\scripts\install.ps1 -Mode Docker -CpuOnly # Docker without GPU
  .\scripts\install.ps1 -Mode Bare            # Python on host

"@
}

function Test-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Err "Docker is not installed."
        Write-Info "Install Docker Desktop from: https://docs.docker.com/desktop/install/windows-install/"
        return $false
    }

    $composeVersion = (docker compose version 2>$null)
    if (-not $composeVersion) {
        Write-Err "Docker Compose v2 not available."
        return $false
    }

    Write-Ok "Docker available"
    return $true
}

function Test-Gpu {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $null = nvidia-smi 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "NVIDIA GPU detected"
                return $true
            }
        } catch {}
    }
    Write-Warn "No NVIDIA GPU detected (CPU-only mode will be used)"
    return $false
}

function Test-Python {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Err "Python is not installed."
        Write-Info "Install Python 3.11+ from: https://www.python.org/downloads/"
        return $false
    }

    $pyVer = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    $parts = $pyVer.Split('.')
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Err "Python 3.10+ required (found: $pyVer)"
        return $false
    }

    Write-Ok "Python $pyVer available"
    return $true
}

function Install-Docker {
    Write-Info "Building Docker images (this may take 10-15 minutes on first run)..."

    Push-Location $RepoRoot
    try {
        if ($CpuOnly) {
            Write-Info "Using CPU-only compose overlay"
            docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml build
            if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
            docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d
        } else {
            docker compose build
            if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
            docker compose up -d
        }

        Write-Info "Waiting for services to become healthy (up to 90 seconds)..."
        for ($i = 1; $i -le 18; $i++) {
            $status = docker compose ps 2>$null
            if ($status -match "\(healthy\)") { break }
            Start-Sleep -Seconds 5
        }

        Write-Host ""
        docker compose ps

        Write-Ok "Docker install complete"
        Write-Info "API:         http://localhost:8000"
        Write-Info "Coordinator: http://localhost:8001"
        Write-Info "Logs:        docker compose logs -f"
        Write-Info "Stop:        docker compose down"
    } finally {
        Pop-Location
    }
}

function Install-Bare {
    Write-Info "Installing Python dependencies..."

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path ".venv")) {
            Write-Info "Creating virtual environment..."
            python -m venv .venv
        }

        & .\.venv\Scripts\python.exe -m pip install --upgrade pip
        & .\.venv\Scripts\python.exe -m pip install -r requirements.txt

        if ($CpuOnly) {
            Write-Info "Pre-downloading models (CPU-only)..."
            & .\.venv\Scripts\python.exe download_models.py --cpu-only
        } else {
            Write-Info "Pre-downloading models..."
            & .\.venv\Scripts\python.exe download_models.py
        }

        Write-Ok "Bare-metal install complete"
        Write-Info "Activate venv: .\.venv\Scripts\Activate.ps1"
        Write-Info "Run pipeline:  python ocr_gpu_async.py"
        Write-Info "Run API:       uvicorn api.main:app --host 0.0.0.0 --port 8000"
    } finally {
        Pop-Location
    }
}

# Main
Write-Banner

if ($Help) {
    Show-Help
    exit 0
}

if (-not $Mode) {
    Write-Host ""
    Write-Info "Choose install mode:"
    Write-Host "  1) Docker (recommended)"
    Write-Host "  2) Bare-metal Python"
    Write-Host ""
    $choice = Read-Host "Selection [1]"
    if (-not $choice) { $choice = "1" }
    switch ($choice) {
        "1" { $Mode = "Docker" }
        "2" { $Mode = "Bare" }
        default { Write-Err "Invalid selection"; exit 1 }
    }
}

if (-not $CpuOnly) {
    if (-not (Test-Gpu)) { $CpuOnly = $true }
}

switch ($Mode) {
    "Docker" {
        if (-not (Test-Docker)) { Write-Err "Docker prerequisites not met"; exit 1 }
        Install-Docker
    }
    "Bare" {
        if (-not (Test-Python)) { Write-Err "Python prerequisites not met"; exit 1 }
        Install-Bare
    }
}

Write-Host ""
Write-Ok "EDCOCR $EdcocrVersion installed"
Write-Host ""
Write-Info "Next steps:"
Write-Host "  - Drop documents in:  $RepoRoot\ocr_source\"
Write-Host "  - Results land in:     $RepoRoot\ocr_output\EXPORT\"
Write-Host "  - Read INSTALL.md for verification steps"
Write-Host "  - Read docs\02-QUICKSTART-5-MINUTE-SUCCESS.md for a guided walkthrough"
Write-Host ""
