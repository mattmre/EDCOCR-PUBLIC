"""ES-9 — Multi-server farm architecture charter (structured contract).

Defines the **charter** for enterprise farm topology: queue master, pool
analyzer, scheduler, cross-server agents, and failure domains.

This module is the **architecture contract**, not a claim that multi-host
farm is live. Implementation lanes:

  ES-10  multi-host registration, capability pools, heartbeat, drain
  ES-11  cross-server routing + no double-commit + custody continuity
  ES-12  farm scale-test vs ES-5
  ES-13  ops runbook
  ES-14  full re-cert + SUCCESS package

Honest residual: single-broker multi-worker compose is **not** multi-server
farm readiness (see enterprise omega narrative).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CHARTER_VERSION = "1.0.0"
CHARTER_ID = "ocr-local-farm-charter-es9"


@dataclass(frozen=True)
class FarmComponent:
    """One architectural component of the multi-server farm."""

    component_id: str
    name: str
    role: str
    responsibilities: tuple[str, ...]
    interfaces: tuple[str, ...]
    failure_modes: tuple[str, ...]
    es_implementation: str  # target implementation lane
    existing_surfaces: tuple[str, ...] = ()  # code that partially covers this


@dataclass(frozen=True)
class FailureDomain:
    """Bounded blast radius for farm failures."""

    domain_id: str
    name: str
    scope: str
    isolation_strategy: str
    recovery: str


@dataclass(frozen=True)
class FarmArchitectureCharter:
    """Top-level farm charter document (structured)."""

    charter_id: str
    version: str
    components: tuple[FarmComponent, ...]
    failure_domains: tuple[FailureDomain, ...]
    non_goals: tuple[str, ...]
    success_criteria_es9: tuple[str, ...]
    implementation_order: tuple[str, ...]
    honest_limits: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples → lists for JSON
        for key in (
            "non_goals",
            "success_criteria_es9",
            "implementation_order",
            "honest_limits",
            "notes",
        ):
            d[key] = list(d[key])
        d["components"] = [
            {
                **c,
                "responsibilities": list(c["responsibilities"]),
                "interfaces": list(c["interfaces"]),
                "failure_modes": list(c["failure_modes"]),
                "existing_surfaces": list(c["existing_surfaces"]),
            }
            for c in d["components"]
        ]
        d["failure_domains"] = list(d["failure_domains"])
        return d

    def component_ids(self) -> list[str]:
        return [c.component_id for c in self.components]

    def validate(self) -> list[str]:
        """Return list of structural validation errors (empty = ok)."""
        errs: list[str] = []
        if self.charter_id != CHARTER_ID:
            errs.append(f"charter_id mismatch: {self.charter_id}")
        if not self.version:
            errs.append("version empty")
        required = {
            "queue_master",
            "pool_analyzer",
            "scheduler",
            "cross_server_agent",
            "failure_domain_control",
        }
        have = set(self.component_ids())
        missing = required - have
        if missing:
            errs.append(f"missing required components: {sorted(missing)}")
        if len(self.failure_domains) < 3:
            errs.append("need >=3 failure domains")
        if "ES-10" not in self.implementation_order:
            errs.append("implementation_order must include ES-10")
        for c in self.components:
            if not c.responsibilities:
                errs.append(f"{c.component_id}: empty responsibilities")
            if not c.interfaces:
                errs.append(f"{c.component_id}: empty interfaces")
            if not c.es_implementation.startswith("ES-"):
                errs.append(f"{c.component_id}: es_implementation must be ES-*")
        # non-goals must forbid reclassifying single-host as farm
        joined = " ".join(self.non_goals).lower()
        if "single-host" not in joined and "single broker" not in joined:
            errs.append("non_goals must call out single-host/broker != farm")
        return errs


def build_default_charter() -> FarmArchitectureCharter:
    """Canonical ES-9 charter content (source of truth for validators/docs)."""
    components = (
        FarmComponent(
            component_id="queue_master",
            name="Queue Master",
            role="Authoritative job admission and queue routing control plane",
            responsibilities=(
                "Accept job submissions and assign document/page work units",
                "Own queue topology (ocr_gpu, ocr_cpu, nlp, compression, …)",
                "Enforce tenant/policy admission and capacity back-pressure",
                "Expose drain and pause for rolling upgrades",
            ),
            interfaces=(
                "REST/API job submit+status (api/, coordinator/jobs/views)",
                "Celery broker queues (RabbitMQ) via coordinator Celery app",
                "Storage backend NFS/S3 for work artifacts",
            ),
            failure_modes=(
                "Broker partition: new submits fail closed or queue locally",
                "Master process crash: durable job state in PostgreSQL",
                "Split-brain: single active master lease (ES-10 heartbeat)",
            ),
            es_implementation="ES-10",
            existing_surfaces=(
                "coordinator/jobs/models.py:Job",
                "coordinator/coordinator/celery.py",
                "api/job_manager.py",
            ),
        ),
        FarmComponent(
            component_id="pool_analyzer",
            name="Pool Analyzer",
            role="Capability-aware worker pool inventory and health scoring",
            responsibilities=(
                "Track online workers by capability (gpu_ocr, cpu_ocr, nlp, …)",
                "Compute pool depth and saturation for routing decisions",
                "Detect stale heartbeats and mark workers offline",
                "Feed auto-tune / load profile recommendations (ES-1/ES-8)",
            ),
            interfaces=(
                "Worker registry (coordinator/jobs/models.py:Worker)",
                "Fleet status API (api/fleet_status.py)",
                "Prometheus metrics (jobs/prometheus_metrics.py)",
            ),
            failure_modes=(
                "Stale registry: workers expire on heartbeat TTL",
                "False offline: recovery on next heartbeat",
                "Capability lie: admission re-checks on task start",
            ),
            es_implementation="ES-10",
            existing_surfaces=(
                "coordinator/jobs/models.py:Worker",
                "api/fleet_status.py",
                "ocr_local/infra/hardware_autotune.py",
            ),
        ),
        FarmComponent(
            component_id="scheduler",
            name="Scheduler / Router",
            role="Cross-host work placement and requeue on failure",
            responsibilities=(
                "Route tasks to queues matching capability and load",
                "Requeue on worker crash without double-commit (ES-11)",
                "Honor priority and tenant fairness",
                "Support per-GPU queue affinity when enabled",
            ),
            interfaces=(
                "Celery task routing (_route_task callable)",
                "OCR_TASK_ROUTING env (gpu/cpu/auto)",
                "PageResult / job state transitions",
            ),
            failure_modes=(
                "Poison task: max retries then DLQ/fail job",
                "Queue lag: back-pressure on admit",
                "Misroute: capability check fails task → requeue",
            ),
            es_implementation="ES-11",
            existing_surfaces=(
                "coordinator/coordinator/celery.py",
                "coordinator/jobs/tasks.py",
                "scripts/vtm_es6_soak_harness.py",
            ),
        ),
        FarmComponent(
            component_id="cross_server_agent",
            name="Cross-Server Agent",
            role="Per-host agent that registers, heartbeats, executes, drains",
            responsibilities=(
                "Register host identity + capabilities with queue master",
                "Heartbeat and report load/VRAM/CPU",
                "Execute assigned OCR/NLP work with forensic custody events",
                "Graceful drain: stop admit, finish in-flight, deregister",
            ),
            interfaces=(
                "Worker registration API / Celery worker boot hooks",
                "mTLS worker channel (when enabled)",
                "Local OCR pipeline entry (ocr_gpu_async / worker image)",
            ),
            failure_modes=(
                "Host crash: in-flight pages requeued by master (ES-11)",
                "Network partition: heartbeat miss → offline + requeue",
                "Agent hang: heal/kill policy (PG-8 class) local to host",
            ),
            es_implementation="ES-10",
            existing_surfaces=(
                "coordinator/docker-compose.worker.yml",
                "coordinator/jobs/tasks.py",
                "docs/integrations/edcommander-sidecar-peer-kit.md",
            ),
        ),
        FarmComponent(
            component_id="failure_domain_control",
            name="Failure Domain Control",
            role="Define blast radius and recovery for farm partitions",
            responsibilities=(
                "Document host, rack/AZ, broker, and storage domains",
                "Specify no double-commit and custody continuity rules (ES-11)",
                "Define join/leave and upgrade order (ES-13)",
            ),
            interfaces=(
                "Charter failure_domains list",
                "Ops runbook (ES-13)",
                "Custody chain continuity checks",
            ),
            failure_modes=(
                "Domain outage: work migrates only if state is durable",
                "Cross-domain split: fail closed on ambiguous ownership",
            ),
            es_implementation="ES-13",
            existing_surfaces=(
                "docs/FAILOVER-RUNBOOK.md",
                "ocr_local/features/custody.py",
            ),
        ),
    )

    domains = (
        FailureDomain(
            domain_id="host",
            name="Single host / worker node",
            scope="One machine running one or more Celery workers",
            isolation_strategy="Heartbeat TTL offline; requeue unfinished pages",
            recovery="Restart agent; re-register; resume from PageResult state",
        ),
        FailureDomain(
            domain_id="broker",
            name="Message broker (RabbitMQ)",
            scope="Queue bus for task dispatch",
            isolation_strategy="Quorum queues + HA overlay when enabled",
            recovery="Broker restore; workers reconnect; no silent ack loss",
        ),
        FailureDomain(
            domain_id="control_plane",
            name="Coordinator control plane (Django/API/DB)",
            scope="Job state, worker registry, metrics",
            isolation_strategy="PostgreSQL durable state; API fail closed on DB loss",
            recovery="Restore DB; workers re-heartbeat; jobs remain queryable",
        ),
        FailureDomain(
            domain_id="storage",
            name="Artifact storage (NFS/S3)",
            scope="Source, temp pages, EXPORT artifacts",
            isolation_strategy="Checksum verification; resume-safe temp dirs",
            recovery="Remount/restore; reprocess only missing pages",
        ),
    )

    return FarmArchitectureCharter(
        charter_id=CHARTER_ID,
        version=CHARTER_VERSION,
        components=components,
        failure_domains=domains,
        non_goals=(
            "Single-host multi-worker docker-compose is NOT multi-server farm readiness",
            "Helm YAML alone is not farm proof",
            "SIDECAR peer kit (ES-3) is peer-app contract, not farm queue master",
            "This charter does not flip production farm features on by default",
        ),
        success_criteria_es9=(
            "Structured charter with queue_master, pool_analyzer, scheduler, "
            "cross_server_agent, failure_domain_control",
            ">=3 failure domains documented with isolation + recovery",
            "Implementation order ES-10…ES-14 declared",
            "Validator script proves charter structural completeness",
            "Honest residual: multi-host not claimed until ES-10…ES-12 evidence",
        ),
        implementation_order=(
            "ES-10",
            "ES-11",
            "ES-12",
            "ES-13",
            "ES-14",
        ),
        honest_limits=(
            "ES-9 is architecture only — no multi-host registration executed here",
            "Existing Worker/Celery surfaces are partial coverage, not farm done",
            "Live multi-VM farm proof is ES-12; ops runbook is ES-13",
        ),
        notes=(
            "Aligns with coordinator multi-capability workers (Phase M3) as the "
            "seed surface for farm expansion",
            "Load profiles (ES-8) remain single-host planning until farm scale-test",
        ),
    )


def charter_coverage_against_repo(
    charter: FarmArchitectureCharter | None = None,
) -> dict[str, Any]:
    """Check that declared existing_surfaces paths exist (honest coverage map)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    c = charter or build_default_charter()
    coverage: dict[str, Any] = {}
    for comp in c.components:
        present = []
        missing = []
        for rel in comp.existing_surfaces:
            # allow "path:symbol" form
            path_part = rel.split(":")[0]
            p = root / path_part
            if p.is_file() or p.is_dir():
                present.append(rel)
            else:
                missing.append(rel)
        coverage[comp.component_id] = {
            "present": present,
            "missing": missing,
            "ok": len(missing) == 0 and len(present) > 0,
        }
    return coverage
