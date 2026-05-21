"""Management command to detect runtime configuration drift.

Captures current environment variable values and compares them against
a known-good baseline snapshot.  Monitors only OCR/Django/Celery/S3/API-
relevant variables (not the full environment).

Usage:
    # Save a baseline snapshot
    python manage.py validate_runtime_config --snapshot config-baseline.json

    # Check for drift against baseline
    python manage.py validate_runtime_config --baseline config-baseline.json

    # Output as JSON
    python manage.py validate_runtime_config --baseline config-baseline.json --json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Monitored variable prefixes and exact names
# ---------------------------------------------------------------------------

MONITORED_PREFIXES: tuple[str, ...] = (
    "DJANGO_",
    "CELERY_",
    "RABBITMQ_",
    "S3_",
    "API_",
    "MAX_",
    "WEBHOOK_",
    "ENABLE_",
    "NUM_",
    "CHUNK_",
    "REDIS_",
    "JOB_",
    "METRICS_",
    "TENANT_",
)

MONITORED_EXACT_NAMES: frozenset[str] = frozenset({
    "DATABASE_URL",
    "DEPLOYMENT_ENV",
    "PRODUCTION_READINESS_ACK",
    "STORAGE_BACKEND",
    "NFS_ROOT",
    "OCR_API_KEY",
    "DPI",
    "TEMP_FOLDER",
    "LOG_DIR",
})


def _is_monitored(name: str) -> bool:
    """Return True if *name* should be tracked for drift detection."""
    if name in MONITORED_EXACT_NAMES:
        return True
    for prefix in MONITORED_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def capture_monitored_env() -> dict[str, str]:
    """Return a dict of current env vars that match monitored filters."""
    return {k: v for k, v in sorted(os.environ.items()) if _is_monitored(k)}


def save_snapshot(filepath: str | Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """Save current monitored env vars as a JSON baseline snapshot.

    Returns the captured environment dict.
    """
    if env is None:
        env = capture_monitored_env()

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variables": env,
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return env


def load_baseline(filepath: str | Path) -> dict[str, str]:
    """Load a previously-saved baseline snapshot and return the variables dict."""
    path = Path(filepath)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("variables", data)


def detect_drift(
    baseline: dict[str, str],
    current: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Compare *baseline* against *current* and categorise differences.

    Returns a dict with keys ``missing``, ``changed``, ``added``, each
    containing a list of detail dicts.
    """
    missing: list[dict[str, str]] = []
    changed: list[dict[str, str]] = []
    added: list[dict[str, str]] = []

    for key in sorted(baseline):
        if key not in current:
            missing.append({"key": key, "baseline_value": baseline[key]})
        elif current[key] != baseline[key]:
            changed.append({
                "key": key,
                "baseline_value": baseline[key],
                "current_value": current[key],
            })

    for key in sorted(current):
        if key not in baseline:
            added.append({"key": key, "current_value": current[key]})

    return {"missing": missing, "changed": changed, "added": added}


def format_table(drift: dict[str, list[dict[str, str]]]) -> str:
    """Format drift results as a human-readable table."""
    lines: list[str] = []

    has_drift = any(drift[cat] for cat in ("missing", "changed", "added"))

    if not has_drift:
        lines.append("No configuration drift detected.")
        return "\n".join(lines)

    if drift["missing"]:
        lines.append("MISSING (in baseline but not in current environment):")
        lines.append(f"  {'Variable':<40s} {'Baseline Value'}")
        lines.append(f"  {'-' * 40} {'-' * 30}")
        for item in drift["missing"]:
            lines.append(f"  {item['key']:<40s} {item['baseline_value']}")
        lines.append("")

    if drift["changed"]:
        lines.append("CHANGED (value differs from baseline):")
        lines.append(f"  {'Variable':<40s} {'Baseline':<30s} {'Current'}")
        lines.append(f"  {'-' * 40} {'-' * 30} {'-' * 30}")
        for item in drift["changed"]:
            lines.append(
                f"  {item['key']:<40s} {item['baseline_value']:<30s} {item['current_value']}"
            )
        lines.append("")

    if drift["added"]:
        lines.append("ADDED (present now but not in baseline):")
        lines.append(f"  {'Variable':<40s} {'Current Value'}")
        lines.append(f"  {'-' * 40} {'-' * 30}")
        for item in drift["added"]:
            lines.append(f"  {item['key']:<40s} {item['current_value']}")
        lines.append("")

    total = sum(len(drift[c]) for c in ("missing", "changed", "added"))
    lines.append(
        f"Summary: {len(drift['missing'])} missing, "
        f"{len(drift['changed'])} changed, "
        f"{len(drift['added'])} added "
        f"({total} total drift items)"
    )

    return "\n".join(lines)


def format_json(drift: dict[str, list[dict[str, str]]]) -> str:
    """Format drift results as a JSON string."""
    has_drift = any(drift[cat] for cat in ("missing", "changed", "added"))
    output = {
        "drift_detected": has_drift,
        "summary": {
            "missing": len(drift["missing"]),
            "changed": len(drift["changed"]),
            "added": len(drift["added"]),
        },
        "details": drift,
    }
    return json.dumps(output, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Django management command wrapper
# ---------------------------------------------------------------------------

try:
    from django.core.management.base import BaseCommand

    class Command(BaseCommand):
        help = "Detect runtime configuration drift against a known-good baseline"

        def add_arguments(self, parser):
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument(
                "--snapshot",
                type=str,
                metavar="FILE",
                help="Save current environment as a baseline JSON snapshot",
            )
            group.add_argument(
                "--baseline",
                type=str,
                metavar="FILE",
                help="Path to baseline JSON snapshot to compare against",
            )
            parser.add_argument(
                "--json",
                action="store_true",
                dest="json_output",
                help="Output results as structured JSON instead of a table",
            )

        def handle(self, *args, **options):
            snapshot_path = options.get("snapshot")
            baseline_path = options.get("baseline")
            use_json = options.get("json_output", False)

            if snapshot_path:
                env = save_snapshot(snapshot_path)
                self.stdout.write(
                    f"Snapshot saved to {snapshot_path} ({len(env)} variables captured)"
                )
                return

            # Baseline comparison mode
            try:
                baseline = load_baseline(baseline_path)
            except FileNotFoundError:
                self.stderr.write(f"ERROR: Baseline file not found: {baseline_path}")
                sys.exit(1)
            except json.JSONDecodeError as exc:
                self.stderr.write(f"ERROR: Invalid JSON in baseline file: {exc}")
                sys.exit(1)

            current = capture_monitored_env()
            drift = detect_drift(baseline, current)

            if use_json:
                self.stdout.write(format_json(drift))
            else:
                self.stdout.write(format_table(drift))

            has_drift = any(drift[cat] for cat in ("missing", "changed", "added"))
            if has_drift:
                sys.exit(1)

except ImportError:
    # Django not available — module can still be used as a library or
    # invoked directly via CLI.
    pass


# ---------------------------------------------------------------------------
# Standalone CLI entry-point (works without Django)
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    """Standalone CLI that works without Django installed."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect runtime configuration drift against a known-good baseline.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snapshot",
        type=str,
        metavar="FILE",
        help="Save current environment as a baseline JSON snapshot",
    )
    group.add_argument(
        "--baseline",
        type=str,
        metavar="FILE",
        help="Path to baseline JSON snapshot to compare against",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as structured JSON instead of a table",
    )

    args = parser.parse_args(argv)

    if args.snapshot:
        env = save_snapshot(args.snapshot)
        print(f"Snapshot saved to {args.snapshot} ({len(env)} variables captured)")
        return 0

    # Baseline comparison
    try:
        baseline = load_baseline(args.baseline)
    except FileNotFoundError:
        print(f"ERROR: Baseline file not found: {args.baseline}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in baseline file: {exc}", file=sys.stderr)
        return 1

    current = capture_monitored_env()
    drift = detect_drift(baseline, current)

    if args.json_output:
        print(format_json(drift))
    else:
        print(format_table(drift))

    has_drift = any(drift[cat] for cat in ("missing", "changed", "added"))
    return 1 if has_drift else 0


if __name__ == "__main__":
    sys.exit(cli_main())
