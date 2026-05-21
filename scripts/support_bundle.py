#!/usr/bin/env python3
"""Support bundle collector for EDCOCR.

Collects system information, configuration, dependencies, logs, health
status, and git state into a timestamped bundle directory for
troubleshooting and support ticket filing.

Security:
    - Environment variable values matching KEY, PASSWORD, SECRET, TOKEN
      patterns are always replaced with ***REDACTED***.
    - .env file values are never included -- only key names are listed.
    - Log files may contain paths but never credentials.

Usage:
    python scripts/support_bundle.py
    python scripts/support_bundle.py --output-dir /tmp/bundles
    python scripts/support_bundle.py --include-logs --include-health
    python scripts/support_bundle.py --json-summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: locate project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SUBPROCESS_TIMEOUT = 10  # seconds

# Patterns that indicate a secret value (case-insensitive check on key name)
_SECRET_PATTERNS = ("KEY", "PASSWORD", "SECRET", "TOKEN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_secret_key(key: str) -> bool:
    """Return True if the env var key name looks like a secret."""
    upper = key.upper()
    return any(pat in upper for pat in _SECRET_PATTERNS)


def _mask_value(key: str, value: str) -> str:
    """Mask the value if the key matches a secret pattern."""
    if _is_secret_key(key):
        return "***REDACTED***"
    return value


def _run_command(
    cmd: list[str],
    *,
    timeout: int = SUBPROCESS_TIMEOUT,
    cwd: str | Path | None = None,
) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined stdout+stderr).

    Never raises -- returns (-1, error_message) on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr
        return result.returncode, output.strip()
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as exc:
        return -1, f"Error running {' '.join(cmd)}: {exc}"


def _safe_disk_usage(path: Path) -> dict | None:
    """Return disk usage for a path, or None if unavailable."""
    try:
        if not path.exists():
            return None
        usage = shutil.disk_usage(str(path))
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def collect_system_info() -> dict:
    """Collect system information (Python, OS, memory, disk, GPU)."""
    info: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "os_release": {},
        "memory": {},
        "disk": {},
        "gpu": {},
    }

    # OS release
    try:
        if hasattr(platform, "freedesktop_os_release"):
            info["os_release"] = platform.freedesktop_os_release()
        else:
            info["os_release"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            }
    except Exception:
        info["os_release"] = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        }

    # Memory (best-effort)
    try:
        if platform.system() == "Linux" and hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            total_bytes = page_size * pages
            info["memory"] = {"total_gb": round(total_bytes / (1024**3), 2)}
        elif platform.system() == "Windows":
            rc, out = _run_command(
                ["wmic", "OS", "get", "TotalVisibleMemorySize", "/value"]
            )
            if rc == 0 and "=" in out:
                kb_str = out.split("=")[-1].strip()
                if kb_str.isdigit():
                    info["memory"] = {
                        "total_gb": round(int(kb_str) / (1024**2), 2)
                    }
        else:
            info["memory"] = {"note": "Memory detection not supported on this platform"}
    except Exception as exc:
        info["memory"] = {"error": str(exc)}

    # Disk space for key directories
    for name, rel in [
        ("ocr_source", "ocr_source"),
        ("ocr_output", "ocr_output"),
        ("ocr_temp", "ocr_temp"),
    ]:
        usage = _safe_disk_usage(PROJECT_ROOT / rel)
        if usage:
            info["disk"][name] = usage

    # GPU info (nvidia-smi)
    rc, gpu_out = _run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if rc == 0:
        info["gpu"] = {"nvidia_smi": gpu_out}
    else:
        info["gpu"] = {"available": False, "note": gpu_out}

    return info


def collect_config_snapshot() -> dict:
    """Collect configuration: env vars (masked), .env keys, compose files, Helm."""
    config: dict = {
        "env_vars": {},
        "coordinator_env_keys": [],
        "docker_compose_files": [],
        "helm_values_exists": False,
    }

    # OCR_*, ENABLE_*, DEPLOYMENT_* env vars with secret masking
    for key, value in sorted(os.environ.items()):
        if key.startswith(("OCR_", "ENABLE_", "DEPLOYMENT_")):
            config["env_vars"][key] = _mask_value(key, value)

    # coordinator/.env keys only (no values)
    env_file = PROJECT_ROOT / "coordinator" / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key_name = line.split("=", 1)[0].strip()
                    config["coordinator_env_keys"].append(key_name)
        except Exception:
            pass

    # Docker compose files
    for pattern in ["docker-compose*.yml", "coordinator/docker-compose*.yml"]:
        matches = glob.glob(str(PROJECT_ROOT / pattern))
        config["docker_compose_files"].extend(
            [str(Path(m).relative_to(PROJECT_ROOT)) for m in sorted(matches)]
        )

    # Helm values.yaml
    helm_values = PROJECT_ROOT / "helm" / "ocr-local" / "values.yaml"
    config["helm_values_exists"] = helm_values.exists()

    return config


def collect_dependency_versions() -> str:
    """Collect pip freeze, docker version, docker-compose version, kubectl version."""
    sections: list[str] = []

    # pip freeze
    rc, out = _run_command([sys.executable, "-m", "pip", "freeze", "--local"])
    sections.append("=== pip freeze ===")
    sections.append(out if rc == 0 else f"Error: {out}")

    # Docker version
    rc, out = _run_command(["docker", "--version"])
    sections.append("\n=== Docker ===")
    sections.append(out if rc == 0 else f"Not available: {out}")

    # docker-compose version (try both forms)
    rc, out = _run_command(["docker", "compose", "version"])
    if rc != 0:
        rc, out = _run_command(["docker-compose", "--version"])
    sections.append("\n=== Docker Compose ===")
    sections.append(out if rc == 0 else f"Not available: {out}")

    # kubectl version
    rc, out = _run_command(["kubectl", "version", "--client", "--short"])
    if rc != 0:
        # Try without --short (newer kubectl removed it)
        rc, out = _run_command(["kubectl", "version", "--client"])
    sections.append("\n=== kubectl ===")
    sections.append(out if rc == 0 else f"Not available: {out}")

    return "\n".join(sections)


def collect_log_tail() -> str:
    """Collect tail of most recent pipeline log and docker container logs."""
    sections: list[str] = []

    # Most recent pipeline log
    log_dir = PROJECT_ROOT / "ocr_output" / "logs"
    if log_dir.exists():
        log_files = sorted(log_dir.glob("ocr_pipeline_*.log"), key=os.path.getmtime)
        if log_files:
            latest = log_files[-1]
            sections.append(f"=== {latest.name} (last 100 lines) ===")
            try:
                lines = latest.read_text(errors="replace").splitlines()
                sections.extend(lines[-100:])
            except Exception as exc:
                sections.append(f"Error reading log: {exc}")
        else:
            sections.append("=== No pipeline logs found ===")
    else:
        sections.append("=== Log directory not found ===")

    # Docker container logs
    sections.append("\n=== Docker container logs (last 50 lines) ===")
    rc, out = _run_command(["docker", "logs", "--tail", "50", "ocr_gpu_processor"])
    if rc == 0:
        sections.append(out)
    else:
        sections.append(f"Not available: {out}")

    return "\n".join(sections)


def collect_health_status() -> dict:
    """Run health check scripts and capture their JSON output."""
    health: dict = {}

    checks = [
        ("verify_release_state", [sys.executable, str(SCRIPT_DIR / "verify_release_state.py"), "--json"]),
        ("validate_topology", [sys.executable, str(SCRIPT_DIR / "validate_topology.py"), "--json"]),
        ("cutover_preflight", [sys.executable, str(SCRIPT_DIR / "cutover_preflight.py"), "--json", "--skip-connectivity"]),
    ]

    for name, cmd in checks:
        rc, out = _run_command(cmd, timeout=30)
        if rc in (0, 1) and out:
            # Try to parse JSON from output (script may print non-JSON preamble)
            try:
                # Find last JSON object in output
                json_start = out.rfind("{")
                if json_start >= 0:
                    health[name] = json.loads(out[json_start:])
                else:
                    health[name] = {"raw_output": out[:2000], "exit_code": rc}
            except json.JSONDecodeError:
                health[name] = {"raw_output": out[:2000], "exit_code": rc}
        else:
            health[name] = {"error": out[:500], "exit_code": rc}

    return health


def collect_git_status() -> str:
    """Collect git branch, commit, dirty state, and recent commits."""
    sections: list[str] = []

    # Current branch and commit
    rc, branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT)
    sections.append(f"Branch: {branch if rc == 0 else 'unknown'}")

    rc, commit = _run_command(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
    sections.append(f"Commit: {commit if rc == 0 else 'unknown'}")

    rc, status = _run_command(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
    if rc == 0:
        dirty = bool(status.strip())
        sections.append(f"Dirty: {dirty}")
        if dirty:
            # Show first 20 lines of changes
            lines = status.strip().splitlines()[:20]
            sections.append("Changed files:")
            sections.extend(f"  {line}" for line in lines)
            if len(status.strip().splitlines()) > 20:
                sections.append(f"  ... and {len(status.strip().splitlines()) - 20} more")

    # Recent 10 commits
    sections.append("\n=== Recent commits ===")
    rc, log_out = _run_command(
        ["git", "log", "--oneline", "-10"], cwd=PROJECT_ROOT
    )
    if rc == 0:
        sections.append(log_out)
    else:
        sections.append(f"Error: {log_out}")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Bundle writer
# ---------------------------------------------------------------------------


def create_bundle(
    *,
    output_dir: Path | None = None,
    include_logs: bool = False,
    include_health: bool = False,
    json_summary: bool = False,
) -> Path:
    """Create the support bundle directory and write all sections.

    Returns the path to the created bundle directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        bundle_dir = Path.cwd() / f"support-bundle-{timestamp}"
    else:
        bundle_dir = Path(output_dir) / f"support-bundle-{timestamp}"

    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []

    def _write_file(name: str, content: str) -> None:
        path = bundle_dir / name
        path.write_text(content, encoding="utf-8")
        manifest_entries.append({
            "file": name,
            "size_bytes": path.stat().st_size,
        })

    # A) System information
    sys_info = collect_system_info()
    _write_file("system.json", json.dumps(sys_info, indent=2))

    # B) Configuration snapshot
    config = collect_config_snapshot()
    _write_file("config.json", json.dumps(config, indent=2))

    # C) Dependency versions
    deps = collect_dependency_versions()
    _write_file("deps.txt", deps)

    # D) Log tail (opt-in)
    if include_logs:
        logs = collect_log_tail()
        _write_file("logs.txt", logs)

    # E) Health status (opt-in)
    if include_health:
        health = collect_health_status()
        _write_file("health.json", json.dumps(health, indent=2))

    # F) Git status
    git_info = collect_git_status()
    _write_file("git.txt", git_info)

    # Manifest — total_files includes the manifest itself
    manifest = {
        "bundle_timestamp": timestamp,
        "bundle_dir": str(bundle_dir),
        "files": manifest_entries,
        "total_files": len(manifest_entries) + 1,  # +1 for manifest.json itself
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Optional JSON summary to stdout
    if json_summary:
        summary = {
            "bundle_dir": str(bundle_dir),
            "timestamp": timestamp,
            "files": [e["file"] for e in manifest_entries],
            "total_size_bytes": sum(e["size_bytes"] for e in manifest_entries),
        }
        print(json.dumps(summary, indent=2))

    return bundle_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Collect a support bundle for EDCOCR troubleshooting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent directory for the bundle (default: current directory)",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        default=False,
        help="Include tail of pipeline logs and Docker container logs",
    )
    parser.add_argument(
        "--include-health",
        action="store_true",
        default=False,
        help="Run health check scripts and include their output",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        default=False,
        help="Print a JSON summary of the bundle to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None

    bundle_dir = create_bundle(
        output_dir=output_dir,
        include_logs=args.include_logs,
        include_health=args.include_health,
        json_summary=args.json_summary,
    )

    if not args.json_summary:
        print(f"Support bundle created: {bundle_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
