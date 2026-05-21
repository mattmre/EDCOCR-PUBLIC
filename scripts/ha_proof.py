#!/usr/bin/env python3
"""HA, autoscaling, and crash-recovery proof framework.

Orchestrates resilience validation through 4 proof categories:
  A) Failover Readiness   - validates drill scripts, runbook, Sentinel, backups
  B) Autoscaling Config   - validates KEDA scalers, resource requests, PDBs
  C) Crash Recovery        - validates resume mechanisms, cleanup commands, errbacks
  D) Infrastructure HA    - validates HA compose, alerts, dashboards, health endpoint

Usage:
    python scripts/ha_proof.py
    python scripts/ha_proof.py --project-root /path/to/repo
    python scripts/ha_proof.py --json
    python scripts/ha_proof.py --report ha_proof_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ocr_local.config.version import __version__
except ImportError:
    __version__ = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A single piece of evidence for a proof check."""

    description: str
    status: str  # "pass", "fail", "warn"
    detail: str = ""


@dataclass
class ProofCategory:
    """Aggregated results for one proof category."""

    name: str
    description: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    status: str = "pending"  # "pass", "fail", "partial"

    def compute_status(self) -> None:
        """Derive overall status from evidence items."""
        if not self.evidence:
            self.status = "fail"
            return
        statuses = [e.status for e in self.evidence]
        if all(s == "pass" for s in statuses):
            self.status = "pass"
        elif any(s == "fail" for s in statuses):
            self.status = "fail"
        else:
            self.status = "partial"


@dataclass
class ProofReport:
    """Complete HA proof report."""

    timestamp: str
    version: str
    categories: list[ProofCategory] = field(default_factory=list)
    verdict: str = "pending"  # "pass", "fail", "partial"

    def compute_verdict(self) -> None:
        """Derive overall verdict from category statuses."""
        for cat in self.categories:
            cat.compute_status()
        statuses = [c.status for c in self.categories]
        if all(s == "pass" for s in statuses):
            self.verdict = "pass"
        elif any(s == "fail" for s in statuses):
            self.verdict = "fail"
        else:
            self.verdict = "partial"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_exists(root: Path, relpath: str) -> bool:
    """Check whether a file exists relative to project root."""
    return (root / relpath).is_file()


def _dir_exists(root: Path, relpath: str) -> bool:
    """Check whether a directory exists relative to project root."""
    return (root / relpath).is_dir()


def _file_contains(root: Path, relpath: str, pattern: str) -> bool:
    """Check whether a file contains text matching a regex pattern."""
    fpath = root / relpath
    if not fpath.is_file():
        return False
    try:
        content = fpath.read_text(encoding="utf-8", errors="replace")
        return bool(re.search(pattern, content))
    except OSError:
        return False


def _glob_files(root: Path, relpath: str, pattern: str) -> list[Path]:
    """Glob for files under a directory relative to project root."""
    target = root / relpath
    if not target.is_dir():
        return []
    return sorted(target.glob(pattern))


# ---------------------------------------------------------------------------
# Proof Category A: Failover Readiness
# ---------------------------------------------------------------------------


def check_failover_readiness(root: Path) -> ProofCategory:
    """Validate failover drill infrastructure and runbook."""
    cat = ProofCategory(
        name="failover_readiness",
        description="Validates failover drill scripts, runbook, Sentinel, and backup configs",
    )

    # A1: failover_drill.py exists and has drill classes
    if _file_exists(root, "scripts/failover_drill.py"):
        has_drill = _file_contains(
            root, "scripts/failover_drill.py", r"class\s+DrillStep"
        )
        if has_drill:
            cat.evidence.append(EvidenceItem(
                description="Failover drill script exists with DrillStep class",
                status="pass",
                detail="scripts/failover_drill.py",
            ))
        else:
            cat.evidence.append(EvidenceItem(
                description="Failover drill script exists but missing DrillStep class",
                status="warn",
                detail="scripts/failover_drill.py found, but DrillStep class not detected",
            ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Failover drill script missing",
            status="fail",
            detail="scripts/failover_drill.py not found",
        ))

    # A2: Failover runbook exists
    if _file_exists(root, "docs/FAILOVER-RUNBOOK.md"):
        has_sections = _file_contains(
            root, "docs/FAILOVER-RUNBOOK.md", r"PostgreSQL Failover"
        )
        cat.evidence.append(EvidenceItem(
            description="Failover runbook exists"
            + (" with PostgreSQL section" if has_sections else ""),
            status="pass" if has_sections else "warn",
            detail="docs/FAILOVER-RUNBOOK.md",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Failover runbook missing",
            status="fail",
            detail="docs/FAILOVER-RUNBOOK.md not found",
        ))

    # A3: Redis Sentinel config in Helm chart
    if _file_exists(root, "helm/ocr-local/templates/redis-sentinel-statefulset.yaml"):
        cat.evidence.append(EvidenceItem(
            description="Redis Sentinel StatefulSet template exists",
            status="pass",
            detail="helm/ocr-local/templates/redis-sentinel-statefulset.yaml",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Redis Sentinel StatefulSet template missing",
            status="fail",
            detail="helm/ocr-local/templates/redis-sentinel-statefulset.yaml not found",
        ))

    # A4: PostgreSQL backup CronJob
    if _file_exists(root, "helm/ocr-local/templates/postgres-backup-cronjob.yaml"):
        cat.evidence.append(EvidenceItem(
            description="PostgreSQL backup CronJob template exists",
            status="pass",
            detail="helm/ocr-local/templates/postgres-backup-cronjob.yaml",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="PostgreSQL backup CronJob template missing",
            status="fail",
            detail="helm/ocr-local/templates/postgres-backup-cronjob.yaml not found",
        ))

    # A5: Baseline evaluation script
    if _file_exists(root, "scripts/evaluate_baseline.py"):
        cat.evidence.append(EvidenceItem(
            description="Baseline evaluation script exists",
            status="pass",
            detail="scripts/evaluate_baseline.py",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Baseline evaluation script missing",
            status="fail",
            detail="scripts/evaluate_baseline.py not found",
        ))

    return cat


# ---------------------------------------------------------------------------
# Proof Category B: Autoscaling Configuration
# ---------------------------------------------------------------------------


def _validate_keda_scaler(root: Path, relpath: str) -> EvidenceItem:
    """Validate a KEDA ScaledObject template has required fields."""
    fname = Path(relpath).name
    if not _file_exists(root, relpath):
        return EvidenceItem(
            description=f"KEDA scaler {fname} missing",
            status="fail",
            detail=f"{relpath} not found",
        )

    content = (root / relpath).read_text(encoding="utf-8", errors="replace")

    required_fields = [
        ("minReplicaCount", r"minReplicaCount"),
        ("maxReplicaCount", r"maxReplicaCount"),
        ("triggers", r"triggers:"),
    ]
    missing = []
    for field_name, pattern in required_fields:
        if not re.search(pattern, content):
            missing.append(field_name)

    if missing:
        return EvidenceItem(
            description=f"KEDA scaler {fname} missing fields: {', '.join(missing)}",
            status="fail",
            detail=relpath,
        )

    return EvidenceItem(
        description=f"KEDA scaler {fname} has required fields",
        status="pass",
        detail=relpath,
    )


def check_autoscaling_config(root: Path) -> ProofCategory:
    """Validate KEDA autoscaler templates and worker resource configuration."""
    cat = ProofCategory(
        name="autoscaling_config",
        description="Validates KEDA scalers, resource requests, and PDB templates",
    )

    # B1: KEDA scaler templates
    keda_scalers = [
        "helm/ocr-local/templates/keda-gpu-scaler.yaml",
        "helm/ocr-local/templates/keda-cpu-scaler.yaml",
        "helm/ocr-local/templates/keda-cpu-ocr-scaler.yaml",
    ]
    for scaler_path in keda_scalers:
        cat.evidence.append(_validate_keda_scaler(root, scaler_path))

    # B2: GPU worker deployment has resource configuration
    gpu_deploy = "helm/ocr-local/templates/gpu-worker-deployment.yaml"
    if _file_exists(root, gpu_deploy):
        has_resources = _file_contains(root, gpu_deploy, r"resources:")
        cat.evidence.append(EvidenceItem(
            description="GPU worker deployment"
            + (" has resource spec" if has_resources else " missing resource spec"),
            status="pass" if has_resources else "fail",
            detail=gpu_deploy,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="GPU worker deployment template missing",
            status="fail",
            detail=f"{gpu_deploy} not found",
        ))

    # B3: CPU worker deployment has resource configuration
    cpu_deploy = "helm/ocr-local/templates/cpu-worker-deployment.yaml"
    if _file_exists(root, cpu_deploy):
        has_resources = _file_contains(root, cpu_deploy, r"resources:")
        cat.evidence.append(EvidenceItem(
            description="CPU worker deployment"
            + (" has resource spec" if has_resources else " missing resource spec"),
            status="pass" if has_resources else "fail",
            detail=cpu_deploy,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="CPU worker deployment template missing",
            status="fail",
            detail=f"{cpu_deploy} not found",
        ))

    # B4: CPU OCR worker deployment has resource configuration
    cpu_ocr_deploy = "helm/ocr-local/templates/cpu-ocr-worker-deployment.yaml"
    if _file_exists(root, cpu_ocr_deploy):
        has_resources = _file_contains(root, cpu_ocr_deploy, r"resources:")
        cat.evidence.append(EvidenceItem(
            description="CPU OCR worker deployment"
            + (" has resource spec" if has_resources else " missing resource spec"),
            status="pass" if has_resources else "fail",
            detail=cpu_ocr_deploy,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="CPU OCR worker deployment template missing",
            status="fail",
            detail=f"{cpu_ocr_deploy} not found",
        ))

    # B5: PDB templates exist with maxUnavailable
    pdb_path = "helm/ocr-local/templates/pdb.yaml"
    if _file_exists(root, pdb_path):
        has_max_unavail = _file_contains(root, pdb_path, r"maxUnavailable:")
        cat.evidence.append(EvidenceItem(
            description="PDB template exists"
            + (" with maxUnavailable configured" if has_max_unavail else ""),
            status="pass" if has_max_unavail else "warn",
            detail=pdb_path,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="PDB template missing",
            status="fail",
            detail=f"{pdb_path} not found",
        ))

    return cat


# ---------------------------------------------------------------------------
# Proof Category C: Crash Recovery
# ---------------------------------------------------------------------------


def check_crash_recovery(root: Path) -> ProofCategory:
    """Validate crash resume mechanisms and cleanup commands."""
    cat = ProofCategory(
        name="crash_recovery",
        description="Validates page-level resume, cleanup commands, and chord errbacks",
    )

    # C1: ocr_gpu_async.py has temp dir handling for crash resume
    pipeline_file = "ocr_gpu_async.py"
    if _file_exists(root, pipeline_file):
        has_temp = _file_contains(root, pipeline_file, r"ocr_temp")
        has_resume = _file_contains(root, pipeline_file, r"TEMP_FOLDER")
        cat.evidence.append(EvidenceItem(
            description="Pipeline has crash resume temp dir handling"
            if (has_temp and has_resume)
            else "Pipeline missing crash resume temp dir handling",
            status="pass" if (has_temp and has_resume) else "fail",
            detail=f"{pipeline_file} (ocr_temp={'found' if has_temp else 'missing'}, "
            f"TEMP_FOLDER={'found' if has_resume else 'missing'})",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Production pipeline file missing",
            status="fail",
            detail=f"{pipeline_file} not found",
        ))

    # C2: cleanup_old_jobs management command
    cleanup_cmd = "coordinator/jobs/management/commands/cleanup_old_jobs.py"
    if _file_exists(root, cleanup_cmd):
        cat.evidence.append(EvidenceItem(
            description="cleanup_old_jobs management command exists",
            status="pass",
            detail=cleanup_cmd,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="cleanup_old_jobs management command missing",
            status="fail",
            detail=f"{cleanup_cmd} not found",
        ))

    # C3: purge_temp_files management command
    purge_cmd = "coordinator/jobs/management/commands/purge_temp_files.py"
    if _file_exists(root, purge_cmd):
        cat.evidence.append(EvidenceItem(
            description="purge_temp_files management command exists",
            status="pass",
            detail=purge_cmd,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="purge_temp_files management command missing",
            status="fail",
            detail=f"{purge_cmd} not found",
        ))

    # C4: Chord errback handling in coordinator tasks
    tasks_file = "coordinator/jobs/tasks.py"
    if _file_exists(root, tasks_file):
        has_errback = _file_contains(root, tasks_file, r"errback")
        has_chord_error = _file_contains(root, tasks_file, r"chord_error_handler")
        cat.evidence.append(EvidenceItem(
            description="Coordinator tasks have chord errback handling"
            if (has_errback and has_chord_error)
            else "Coordinator tasks missing chord errback handling",
            status="pass" if (has_errback and has_chord_error) else "fail",
            detail=f"{tasks_file} (errback={'found' if has_errback else 'missing'}, "
            f"chord_error_handler={'found' if has_chord_error else 'missing'})",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Coordinator tasks file missing",
            status="fail",
            detail=f"{tasks_file} not found",
        ))

    # C5: Scale test crash recovery support
    scale_test = "scale_test.py"
    if _file_exists(root, scale_test):
        has_crash_recovery = _file_contains(
            root, scale_test, r"crash.recovery|crash_recovery"
        )
        cat.evidence.append(EvidenceItem(
            description="Scale test supports crash recovery mode"
            if has_crash_recovery
            else "Scale test exists but crash recovery mode not detected",
            status="pass" if has_crash_recovery else "warn",
            detail=scale_test,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Scale test framework missing",
            status="fail",
            detail=f"{scale_test} not found",
        ))

    return cat


# ---------------------------------------------------------------------------
# Proof Category D: Infrastructure HA Components
# ---------------------------------------------------------------------------


def check_infrastructure_ha(root: Path) -> ProofCategory:
    """Validate HA infrastructure components and monitoring."""
    cat = ProofCategory(
        name="infrastructure_ha",
        description="Validates HA compose, alerts, dashboards, and health endpoint",
    )

    # D1: docker-compose.ha.yml exists and references HA components
    ha_compose = "coordinator/docker-compose.ha.yml"
    if _file_exists(root, ha_compose):
        has_sentinel = _file_contains(root, ha_compose, r"sentinel|redis")
        has_rabbitmq_cluster = _file_contains(root, ha_compose, r"rabbitmq2|rabbitmq3")
        detail_parts = []
        if has_sentinel:
            detail_parts.append("Redis/Sentinel")
        if has_rabbitmq_cluster:
            detail_parts.append("RabbitMQ cluster")
        cat.evidence.append(EvidenceItem(
            description="HA compose overlay exists"
            + (f" with {', '.join(detail_parts)}" if detail_parts else ""),
            status="pass" if (has_sentinel or has_rabbitmq_cluster) else "warn",
            detail=ha_compose,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="HA compose overlay missing",
            status="fail",
            detail=f"{ha_compose} not found",
        ))

    # D2: PrometheusRule template with alert rules
    prom_rule = "helm/ocr-local/templates/prometheusrule.yaml"
    if _file_exists(root, prom_rule):
        has_alerts = _file_contains(root, prom_rule, r"alert:\s+\w+")
        cat.evidence.append(EvidenceItem(
            description="PrometheusRule template exists"
            + (" with alert definitions" if has_alerts else ""),
            status="pass" if has_alerts else "warn",
            detail=prom_rule,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="PrometheusRule template missing",
            status="fail",
            detail=f"{prom_rule} not found",
        ))

    # D3: Grafana dashboard configmap
    grafana_cm = "helm/ocr-local/templates/grafana-dashboard-configmap.yaml"
    if _file_exists(root, grafana_cm):
        cat.evidence.append(EvidenceItem(
            description="Grafana dashboard configmap exists",
            status="pass",
            detail=grafana_cm,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Grafana dashboard configmap missing",
            status="fail",
            detail=f"{grafana_cm} not found",
        ))

    # D4: Health check endpoint
    health_router = "api/routers/health.py"
    if _file_exists(root, health_router):
        has_endpoint = _file_contains(root, health_router, r"/api/v1/health")
        cat.evidence.append(EvidenceItem(
            description="Health check endpoint exists"
            + (" at /api/v1/health" if has_endpoint else ""),
            status="pass" if has_endpoint else "warn",
            detail=health_router,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Health check router missing",
            status="fail",
            detail=f"{health_router} not found",
        ))

    # D5: Healthcheck shell script for Docker
    if _file_exists(root, "healthcheck.sh"):
        cat.evidence.append(EvidenceItem(
            description="Docker healthcheck script exists",
            status="pass",
            detail="healthcheck.sh",
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="Docker healthcheck script missing",
            status="fail",
            detail="healthcheck.sh not found",
        ))

    # D6: ServiceMonitor template for Prometheus scraping
    svc_monitor = "helm/ocr-local/templates/servicemonitor.yaml"
    if _file_exists(root, svc_monitor):
        cat.evidence.append(EvidenceItem(
            description="ServiceMonitor template exists",
            status="pass",
            detail=svc_monitor,
        ))
    else:
        cat.evidence.append(EvidenceItem(
            description="ServiceMonitor template missing",
            status="fail",
            detail=f"{svc_monitor} not found",
        ))

    return cat


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_report(root: Path) -> ProofReport:
    """Run all proof categories and build the full report."""
    report = ProofReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=__version__,
    )
    report.categories.append(check_failover_readiness(root))
    report.categories.append(check_autoscaling_config(root))
    report.categories.append(check_crash_recovery(root))
    report.categories.append(check_infrastructure_ha(root))
    report.compute_verdict()
    return report


def format_text(report: ProofReport) -> str:
    """Format the report as a human-readable text table."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("HA / Autoscaling / Crash-Recovery Proof Report")
    lines.append(f"Generated: {report.timestamp}")
    lines.append(f"Version: {report.version}")
    lines.append("=" * 72)

    for cat in report.categories:
        status_label = cat.status.upper()
        lines.append("")
        lines.append(f"--- {cat.name} [{status_label}] ---")
        lines.append(f"    {cat.description}")
        for ev in cat.evidence:
            marker = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]"}.get(
                ev.status, "[????]"
            )
            lines.append(f"  {marker} {ev.description}")
            if ev.detail:
                lines.append(f"         {ev.detail}")

    lines.append("")
    lines.append("=" * 72)
    verdict_label = report.verdict.upper()
    total_pass = sum(
        1 for c in report.categories for e in c.evidence if e.status == "pass"
    )
    total_fail = sum(
        1 for c in report.categories for e in c.evidence if e.status == "fail"
    )
    total_warn = sum(
        1 for c in report.categories for e in c.evidence if e.status == "warn"
    )
    total = total_pass + total_fail + total_warn
    lines.append(
        f"VERDICT: {verdict_label}  "
        f"({total_pass}/{total} pass, {total_fail} fail, {total_warn} warn)"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def format_json(report: ProofReport) -> str:
    """Format the report as JSON."""
    return json.dumps(asdict(report), indent=2)


def format_markdown(report: ProofReport) -> str:
    """Format the report as a Markdown document."""
    lines: list[str] = []
    lines.append("# HA / Autoscaling / Crash-Recovery Proof Report")
    lines.append("")
    lines.append(f"**Generated**: {report.timestamp}")
    lines.append(f"**Version**: {report.version}")
    lines.append(f"**Verdict**: {report.verdict.upper()}")
    lines.append("")

    for cat in report.categories:
        status_label = cat.status.upper()
        lines.append(f"## {cat.name} [{status_label}]")
        lines.append("")
        lines.append(cat.description)
        lines.append("")
        lines.append("| Status | Check | Detail |")
        lines.append("|--------|-------|--------|")
        for ev in cat.evidence:
            marker = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}.get(
                ev.status, "????"
            )
            detail_escaped = ev.detail.replace("|", "\\|") if ev.detail else ""
            desc_escaped = ev.description.replace("|", "\\|")
            lines.append(f"| {marker} | {desc_escaped} | {detail_escaped} |")
        lines.append("")

    total_pass = sum(
        1 for c in report.categories for e in c.evidence if e.status == "pass"
    )
    total = sum(len(c.evidence) for c in report.categories)
    lines.append("---")
    lines.append("")
    lines.append(f"**Score**: {total_pass}/{total} checks passed")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HA, autoscaling, and crash-recovery proof framework"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write markdown report to file",
    )
    args = parser.parse_args(argv)

    root = args.project_root
    if root is None:
        root = Path(__file__).resolve().parent.parent
    root = root.resolve()

    report = build_report(root)

    if args.json_output:
        print(format_json(report))
    else:
        print(format_text(report))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(format_markdown(report), encoding="utf-8")
        print(f"\nMarkdown report written to {args.report}")

    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
