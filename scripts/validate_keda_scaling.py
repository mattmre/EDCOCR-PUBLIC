"""KEDA autoscaling validation tool.

Validates KEDA ScaledObject configurations from Helm values, simulates
queue depth changes, tests cooldown period calculations, and validates
min/max replica bounds.  Generates a validation report in JSON and
markdown.

Usage:
    python scripts/validate_keda_scaling.py
    python scripts/validate_keda_scaling.py --helm-values helm/ocr-local/values.yaml
    python scripts/validate_keda_scaling.py --output-dir ./reports
    python scripts/validate_keda_scaling.py --simulate-duration 600
"""

import argparse
import datetime
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default cooldown/polling by strategy (mirrors keda-gpu-scaler.yaml)
STRATEGY_DEFAULTS = {
    "aggressive": {"pollingInterval": 10, "cooldownPeriod": 60},
    "balanced": {"pollingInterval": 15, "cooldownPeriod": 300},
    "conservative": {"pollingInterval": 30, "cooldownPeriod": 600},
}

# Maximum allowed by KEDA
KEDA_MAX_REPLICAS_LIMIT = 1000
KEDA_MIN_POLLING_INTERVAL = 5
KEDA_MIN_COOLDOWN_PERIOD = 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScalerConfig:
    """Parsed KEDA scaler configuration for one worker type."""

    name: str = ""
    enabled: bool = False
    min_replicas: int = 0
    max_replicas: int = 0
    polling_interval: int = 15
    cooldown_period: int = 300
    queue_target: int = 5
    strategy: str = "balanced"
    effective_polling: int = 15
    effective_cooldown: int = 300


@dataclass
class ValidationFinding:
    """A single validation finding (pass, warning, or error)."""

    scaler: str = ""
    severity: str = "info"  # "info", "warning", "error"
    message: str = ""


@dataclass
class ScaleSimulationStep:
    """A single step in a scaling simulation."""

    time_s: int = 0
    queue_depth: int = 0
    desired_replicas: int = 0
    actual_replicas: int = 0
    scaling_action: str = "none"  # "none", "scale_up", "scale_down", "cooldown"


@dataclass
class ScaleSimulationResult:
    """Results of a scaling simulation for one scaler."""

    scaler_name: str = ""
    duration_s: int = 0
    steps: list = field(default_factory=list)
    max_replicas_reached: int = 0
    scale_up_events: int = 0
    scale_down_events: int = 0
    avg_response_time_s: float = 0.0
    queue_saturation_pct: float = 0.0


@dataclass
class KedaValidationReport:
    """Complete KEDA validation report."""

    timestamp: str = ""
    helm_values_path: str = ""
    global_strategy: str = ""
    global_max_replicas: int = 0
    scalers: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    simulations: list = field(default_factory=list)
    passed: bool = True
    total_errors: int = 0
    total_warnings: int = 0


# ---------------------------------------------------------------------------
# Helm values parser
# ---------------------------------------------------------------------------


def parse_helm_values(values_path: str = None) -> dict:
    """Parse Helm values.yaml and extract KEDA-related configuration.

    Parameters
    ----------
    values_path : str, optional
        Path to values.yaml. Default: helm/ocr-local/values.yaml.

    Returns
    -------
    dict
        Parsed values dictionary.
    """
    if values_path is None:
        values_path = str(_PROJECT_ROOT / "helm" / "ocr-local" / "values.yaml")

    path = Path(values_path)
    if not path.exists():
        logger.warning("Helm values not found at %s; using built-in defaults", values_path)
        return _get_default_values()

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available; using built-in defaults")
        return _get_default_values()

    try:
        with open(path, encoding="utf-8") as f:
            values = yaml.safe_load(f)
        return values or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s; using defaults", values_path, e)
        return _get_default_values()


def _get_default_values() -> dict:
    """Return built-in default values matching the Helm chart."""
    return {
        "keda": {
            "scalingStrategy": "balanced",
            "maxReplicaCount": 50,
        },
        "gpuWorker": {
            "autoscaling": {
                "enabled": False,
                "minReplicas": 1,
                "maxReplicas": 20,
                "pollingInterval": 15,
                "cooldownPeriod": 300,
                "queueTarget": 5,
            },
        },
        "cpuWorker": {
            "autoscaling": {
                "enabled": False,
                "minReplicas": 1,
                "maxReplicas": 30,
                "pollingInterval": 15,
                "cooldownPeriod": 120,
                "queueTarget": 3,
            },
        },
        "nlpWorker": {
            "autoscaling": {
                "enabled": False,
                "minReplicas": 0,
                "maxReplicas": 10,
                "pollingInterval": 15,
                "cooldownPeriod": 120,
                "queueTarget": 5,
            },
        },
        "layoutlmWorker": {
            "autoscaling": {
                "enabled": False,
                "minReplicas": 0,
                "maxReplicas": 3,
                "pollingInterval": 15,
                "cooldownPeriod": 120,
                "queueTarget": 2,
            },
        },
    }


def extract_scalers(values: dict) -> list:
    """Extract scaler configurations from parsed Helm values.

    Parameters
    ----------
    values : dict
        Parsed Helm values.

    Returns
    -------
    list[ScalerConfig]
        Extracted scaler configs.
    """
    keda_config = values.get("keda", {})
    global_strategy = keda_config.get("scalingStrategy", "balanced")
    global_max = keda_config.get("maxReplicaCount", 50)

    strategy_overrides = STRATEGY_DEFAULTS.get(global_strategy, {})

    worker_keys = [
        ("gpuWorker", "GPU Worker"),
        ("cpuWorker", "CPU Worker"),
        ("nlpWorker", "NLP Worker"),
        ("layoutlmWorker", "LayoutLM Worker"),
    ]

    scalers = []
    for key, name in worker_keys:
        worker = values.get(key, {})
        autoscaling = worker.get("autoscaling", {})

        polling = autoscaling.get("pollingInterval", 15)
        cooldown = autoscaling.get("cooldownPeriod", 300)

        # Apply strategy overrides
        eff_polling = strategy_overrides.get("pollingInterval", polling)
        eff_cooldown = strategy_overrides.get("cooldownPeriod", cooldown)

        max_reps = autoscaling.get("maxReplicas", 10)
        effective_max = min(max_reps, global_max)

        scalers.append(ScalerConfig(
            name=name,
            enabled=autoscaling.get("enabled", False),
            min_replicas=autoscaling.get("minReplicas", 0),
            max_replicas=effective_max,
            polling_interval=polling,
            cooldown_period=cooldown,
            queue_target=autoscaling.get("queueTarget", 5),
            strategy=global_strategy,
            effective_polling=eff_polling,
            effective_cooldown=eff_cooldown,
        ))

    return scalers


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


def validate_scaler(scaler: ScalerConfig) -> list:
    """Validate a single scaler configuration.

    Parameters
    ----------
    scaler : ScalerConfig
        Scaler to validate.

    Returns
    -------
    list[ValidationFinding]
        Findings for this scaler.
    """
    findings = []

    # Min replicas <= Max replicas
    if scaler.min_replicas > scaler.max_replicas:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="error",
            message=(
                f"minReplicas ({scaler.min_replicas}) > "
                f"maxReplicas ({scaler.max_replicas})"
            ),
        ))

    # Max replicas within KEDA limits
    if scaler.max_replicas > KEDA_MAX_REPLICAS_LIMIT:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="error",
            message=(
                f"maxReplicas ({scaler.max_replicas}) exceeds "
                f"KEDA limit ({KEDA_MAX_REPLICAS_LIMIT})"
            ),
        ))

    # Min replicas non-negative
    if scaler.min_replicas < 0:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="error",
            message=f"minReplicas ({scaler.min_replicas}) must be >= 0",
        ))

    # Queue target positive
    if scaler.queue_target <= 0:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="error",
            message=f"queueTarget ({scaler.queue_target}) must be > 0",
        ))

    # Polling interval sanity
    if scaler.effective_polling < KEDA_MIN_POLLING_INTERVAL:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="warning",
            message=(
                f"Effective pollingInterval ({scaler.effective_polling}s) is very low; "
                f"KEDA minimum recommended is {KEDA_MIN_POLLING_INTERVAL}s"
            ),
        ))

    # Cooldown period sanity
    if scaler.effective_cooldown < 30:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="warning",
            message=(
                f"Effective cooldownPeriod ({scaler.effective_cooldown}s) is very short; "
                "may cause excessive scaling churn"
            ),
        ))

    # GPU worker specific: warn if min_replicas=0
    if "GPU" in scaler.name and scaler.min_replicas == 0:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="warning",
            message=(
                "minReplicas=0 for GPU worker means cold starts from zero; "
                "consider minReplicas=1 for latency-sensitive workloads"
            ),
        ))

    # Pass finding if no issues
    if not findings:
        findings.append(ValidationFinding(
            scaler=scaler.name,
            severity="info",
            message="Configuration valid",
        ))

    return findings


# ---------------------------------------------------------------------------
# Scaling simulation
# ---------------------------------------------------------------------------


def simulate_scaling(
    scaler: ScalerConfig,
    duration_s: int = 600,
    workload_pattern: str = "burst",
) -> ScaleSimulationResult:
    """Simulate KEDA scaling behavior over time.

    Parameters
    ----------
    scaler : ScalerConfig
        Scaler configuration.
    duration_s : int
        Simulation duration in seconds.
    workload_pattern : str
        "burst" (spike then decline), "steady" (constant), or "wave" (sinusoidal).

    Returns
    -------
    ScaleSimulationResult
        Simulation results.
    """
    steps = []
    current_replicas = scaler.min_replicas
    last_scale_down_time = -scaler.effective_cooldown  # allow immediate first scale
    scale_up_events = 0
    scale_down_events = 0
    max_replicas_reached = 0
    response_times = []

    for t in range(0, duration_s, scaler.effective_polling):
        # Generate queue depth based on pattern
        queue_depth = _generate_queue_depth(t, duration_s, workload_pattern)

        # KEDA scaling logic: desired = ceil(queue_depth / queue_target)
        if queue_depth > 0:
            desired = min(
                math.ceil(queue_depth / max(1, scaler.queue_target)),
                scaler.max_replicas,
            )
            desired = max(desired, scaler.min_replicas)
        else:
            desired = scaler.min_replicas

        action = "none"

        if desired > current_replicas:
            # Scale up (immediate)
            current_replicas = desired
            action = "scale_up"
            scale_up_events += 1
            response_times.append(scaler.effective_polling)
        elif desired < current_replicas:
            # Scale down (respect cooldown)
            if (t - last_scale_down_time) >= scaler.effective_cooldown:
                current_replicas = desired
                action = "scale_down"
                scale_down_events += 1
                last_scale_down_time = t
            else:
                action = "cooldown"

        max_replicas_reached = max(max_replicas_reached, current_replicas)

        steps.append(ScaleSimulationStep(
            time_s=t,
            queue_depth=queue_depth,
            desired_replicas=desired,
            actual_replicas=current_replicas,
            scaling_action=action,
        ))

    # Queue saturation: fraction of time where queue > target * replicas
    saturated_steps = sum(
        1 for s in steps
        if s.queue_depth > scaler.queue_target * max(1, s.actual_replicas)
    )
    saturation_pct = (saturated_steps / max(1, len(steps))) * 100

    return ScaleSimulationResult(
        scaler_name=scaler.name,
        duration_s=duration_s,
        steps=[asdict(s) for s in steps],
        max_replicas_reached=max_replicas_reached,
        scale_up_events=scale_up_events,
        scale_down_events=scale_down_events,
        avg_response_time_s=round(
            sum(response_times) / max(1, len(response_times)), 2
        ),
        queue_saturation_pct=round(saturation_pct, 2),
    )


def _generate_queue_depth(t: int, duration_s: int, pattern: str) -> int:
    """Generate a simulated queue depth at time t.

    Parameters
    ----------
    t : int
        Current time in seconds.
    duration_s : int
        Total simulation duration.
    pattern : str
        Workload pattern.

    Returns
    -------
    int
        Simulated queue depth.
    """
    if pattern == "burst":
        # Spike at 1/4 duration, then decay
        peak_t = duration_s // 4
        if t < peak_t:
            return int(50 * (t / peak_t))
        else:
            decay = max(0, 1 - (t - peak_t) / (duration_s - peak_t))
            return int(50 * decay)
    elif pattern == "steady":
        return 20
    elif pattern == "wave":
        # Sinusoidal with period = duration/3
        period = max(1, duration_s // 3)
        return int(25 + 25 * math.sin(2 * math.pi * t / period))
    else:
        return 10


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------


def run_keda_validation(
    helm_values_path: str = None,
    simulate_duration: int = 600,
) -> KedaValidationReport:
    """Run the full KEDA scaling validation.

    Parameters
    ----------
    helm_values_path : str, optional
        Path to Helm values.yaml.
    simulate_duration : int
        Duration for scaling simulations in seconds.

    Returns
    -------
    KedaValidationReport
        Complete validation report.
    """
    values = parse_helm_values(helm_values_path)
    keda_config = values.get("keda", {})
    global_strategy = keda_config.get("scalingStrategy", "balanced")
    global_max = keda_config.get("maxReplicaCount", 50)

    scalers = extract_scalers(values)

    # Validate each scaler
    all_findings = []
    for scaler in scalers:
        findings = validate_scaler(scaler)
        all_findings.extend(findings)

    # Count errors and warnings
    total_errors = sum(1 for f in all_findings if f.severity == "error")
    total_warnings = sum(1 for f in all_findings if f.severity == "warning")

    # Simulate scaling for enabled scalers (or all for validation)
    simulations = []
    for scaler in scalers:
        for pattern in ["burst", "steady", "wave"]:
            sim = simulate_scaling(scaler, simulate_duration, pattern)
            sim.scaler_name = f"{scaler.name} ({pattern})"
            simulations.append(sim)

    return KedaValidationReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        helm_values_path=helm_values_path or "default",
        global_strategy=global_strategy,
        global_max_replicas=global_max,
        scalers=[asdict(s) for s in scalers],
        findings=[asdict(f) for f in all_findings],
        simulations=[asdict(s) for s in simulations],
        passed=(total_errors == 0),
        total_errors=total_errors,
        total_warnings=total_warnings,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: KedaValidationReport) -> str:
    """Format the KEDA validation report as markdown.

    Parameters
    ----------
    report : KedaValidationReport
        Complete report.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# KEDA Autoscaling Validation Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**Status**: {status}",
        f"**Strategy**: {report.global_strategy}",
        f"**Global max replicas**: {report.global_max_replicas}",
        f"**Errors**: {report.total_errors}",
        f"**Warnings**: {report.total_warnings}",
        "",
        "## Scaler Configurations",
        "",
        "| Name | Enabled | Min | Max | Polling (s) | Cooldown (s) | Queue Target |",
        "|------|---------|-----|-----|-------------|-------------|-------------|",
    ]

    for s in report.scalers:
        lines.append(
            f"| {s['name']} | {s['enabled']} | {s['min_replicas']} | {s['max_replicas']} "
            f"| {s['effective_polling']} | {s['effective_cooldown']} "
            f"| {s['queue_target']} |"
        )

    lines.extend(["", "## Findings", ""])

    for f in report.findings:
        icon = {"error": "[ERROR]", "warning": "[WARN]", "info": "[OK]"}.get(f["severity"], "")
        lines.append(f"- {icon} **{f['scaler']}**: {f['message']}")

    lines.extend(["", "## Simulation Summary", ""])
    lines.append(
        "| Scaler | Max Replicas | Scale-ups | Scale-downs | Saturation % |"
    )
    lines.append(
        "|--------|-------------|-----------|-------------|-------------|"
    )

    for sim in report.simulations:
        lines.append(
            f"| {sim['scaler_name']} | {sim['max_replicas_reached']} "
            f"| {sim['scale_up_events']} | {sim['scale_down_events']} "
            f"| {sim['queue_saturation_pct']:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def format_console_report(report: KedaValidationReport) -> str:
    """Format the KEDA validation report for console output.

    Parameters
    ----------
    report : KedaValidationReport
        Complete report.

    Returns
    -------
    str
        Console-formatted report.
    """
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "",
        "=" * 90,
        "KEDA AUTOSCALING VALIDATION REPORT",
        "=" * 90,
        "",
        f"  Status:           {status}",
        f"  Strategy:         {report.global_strategy}",
        f"  Global max reps:  {report.global_max_replicas}",
        f"  Errors:           {report.total_errors}",
        f"  Warnings:         {report.total_warnings}",
        "",
        "-" * 90,
        "SCALER CONFIGURATIONS",
        "-" * 90,
        f"{'Name':<20} {'On':>3} {'Min':>4} {'Max':>4} {'Poll(s)':>8} "
        f"{'Cool(s)':>8} {'QTarget':>8}",
        "-" * 90,
    ]

    for s in report.scalers:
        en = "Y" if s["enabled"] else "N"
        lines.append(
            f"{s['name']:<20} {en:>3} {s['min_replicas']:>4} {s['max_replicas']:>4} "
            f"{s['effective_polling']:>8} {s['effective_cooldown']:>8} "
            f"{s['queue_target']:>8}"
        )

    lines.extend([
        "",
        "-" * 90,
        "FINDINGS",
        "-" * 90,
    ])

    for f in report.findings:
        sev = f["severity"].upper()
        lines.append(f"  [{sev:>5}] {f['scaler']}: {f['message']}")

    lines.extend([
        "",
        "-" * 90,
        "SIMULATION SUMMARY",
        "-" * 90,
        f"{'Scaler':<30} {'MaxRep':>6} {'Up':>4} {'Down':>4} {'Sat%':>6}",
        "-" * 90,
    ])

    for sim in report.simulations:
        lines.append(
            f"{sim['scaler_name']:<30} {sim['max_replicas_reached']:>6} "
            f"{sim['scale_up_events']:>4} {sim['scale_down_events']:>4} "
            f"{sim['queue_saturation_pct']:>5.1f}%"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for KEDA scaling validation."""
    parser = argparse.ArgumentParser(
        description="Validate KEDA autoscaling configuration and simulate behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_keda_scaling.py
  python scripts/validate_keda_scaling.py --helm-values helm/ocr-local/values.yaml
  python scripts/validate_keda_scaling.py --simulate-duration 1200
  python scripts/validate_keda_scaling.py --output-dir ./reports
        """,
    )
    parser.add_argument(
        "--helm-values",
        type=str,
        default=None,
        help="Path to Helm values.yaml (default: helm/ocr-local/values.yaml)",
    )
    parser.add_argument(
        "--simulate-duration",
        type=int,
        default=600,
        help="Simulation duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output reports (JSON + markdown)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Running KEDA scaling validation...")

    report = run_keda_validation(
        helm_values_path=args.helm_values,
        simulate_duration=args.simulate_duration,
    )

    # Console output
    print(format_console_report(report))

    # Save reports
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "keda_validation.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        md_path = out_dir / "keda_validation.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(report))
        logger.info("Markdown report saved to %s", md_path)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
