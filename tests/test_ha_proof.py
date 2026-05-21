"""Tests for scripts/ha_proof.py — HA proof framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from ha_proof import (
    EvidenceItem,
    ProofCategory,
    ProofReport,
    build_report,
    check_autoscaling_config,
    check_crash_recovery,
    check_failover_readiness,
    check_infrastructure_ha,
    format_json,
    format_markdown,
    format_text,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_root() -> Path:
    """Return the real project root for integration-style checks."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal fake project tree for isolated tests."""
    # Scripts
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "failover_drill.py").write_text(
        "class DrillStep:\n    pass\n", encoding="utf-8"
    )
    (scripts / "evaluate_baseline.py").write_text("# baseline\n", encoding="utf-8")

    # Docs
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "FAILOVER-RUNBOOK.md").write_text(
        "## PostgreSQL Failover\nSteps...\n", encoding="utf-8"
    )

    # Helm templates
    helm_tpl = tmp_path / "helm" / "ocr-local" / "templates"
    helm_tpl.mkdir(parents=True)
    (helm_tpl / "redis-sentinel-statefulset.yaml").write_text(
        "kind: StatefulSet\n", encoding="utf-8"
    )
    (helm_tpl / "postgres-backup-cronjob.yaml").write_text(
        "kind: CronJob\n", encoding="utf-8"
    )

    # KEDA scalers
    keda_content = (
        "apiVersion: keda.sh/v1alpha1\nkind: ScaledObject\nspec:\n"
        "  minReplicaCount: 1\n  maxReplicaCount: 5\n  triggers:\n"
        "    - type: rabbitmq\n"
    )
    (helm_tpl / "keda-gpu-scaler.yaml").write_text(keda_content, encoding="utf-8")
    (helm_tpl / "keda-cpu-scaler.yaml").write_text(keda_content, encoding="utf-8")
    (helm_tpl / "keda-cpu-ocr-scaler.yaml").write_text(keda_content, encoding="utf-8")

    # Worker deployments
    deploy_content = "kind: Deployment\nspec:\n  resources:\n    requests:\n      cpu: 1\n"
    (helm_tpl / "gpu-worker-deployment.yaml").write_text(
        deploy_content, encoding="utf-8"
    )
    (helm_tpl / "cpu-worker-deployment.yaml").write_text(
        deploy_content, encoding="utf-8"
    )
    (helm_tpl / "cpu-ocr-worker-deployment.yaml").write_text(
        deploy_content, encoding="utf-8"
    )

    # PDB
    (helm_tpl / "pdb.yaml").write_text(
        "kind: PodDisruptionBudget\nspec:\n  maxUnavailable: 1\n",
        encoding="utf-8",
    )

    # PrometheusRule
    (helm_tpl / "prometheusrule.yaml").write_text(
        "spec:\n  groups:\n    - rules:\n        - alert: TestAlert\n",
        encoding="utf-8",
    )

    # Grafana
    (helm_tpl / "grafana-dashboard-configmap.yaml").write_text(
        "kind: ConfigMap\n", encoding="utf-8"
    )

    # ServiceMonitor
    (helm_tpl / "servicemonitor.yaml").write_text(
        "kind: ServiceMonitor\n", encoding="utf-8"
    )

    # Pipeline file
    (tmp_path / "ocr_gpu_async.py").write_text(
        "TEMP_FOLDER = '/app/ocr_temp'\n# resume logic\n", encoding="utf-8"
    )

    # Coordinator management commands
    mgmt = tmp_path / "coordinator" / "jobs" / "management" / "commands"
    mgmt.mkdir(parents=True)
    (mgmt / "cleanup_old_jobs.py").write_text("class Command:\n    pass\n", encoding="utf-8")
    (mgmt / "purge_temp_files.py").write_text("class Command:\n    pass\n", encoding="utf-8")

    # Coordinator tasks
    tasks_dir = tmp_path / "coordinator" / "jobs"
    (tasks_dir / "tasks.py").write_text(
        "errback = chord_error_handler.s(str(job_id))\n", encoding="utf-8"
    )

    # Scale test
    (tmp_path / "scale_test.py").write_text(
        "# crash_recovery mode\n", encoding="utf-8"
    )

    # HA compose
    coord = tmp_path / "coordinator"
    (coord / "docker-compose.ha.yml").write_text(
        "services:\n  rabbitmq2:\n    image: rabbitmq\n  sentinel:\n    image: redis\n",
        encoding="utf-8",
    )

    # API health router
    api_routers = tmp_path / "api" / "routers"
    api_routers.mkdir(parents=True)
    (api_routers / "health.py").write_text(
        "@router.get('/api/v1/health')\ndef health(): pass\n", encoding="utf-8"
    )

    # Healthcheck script
    (tmp_path / "healthcheck.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_create_pass(self):
        ev = EvidenceItem("test check", "pass", "some detail")
        assert ev.status == "pass"
        assert ev.description == "test check"
        assert ev.detail == "some detail"

    def test_create_fail(self):
        ev = EvidenceItem("failing check", "fail")
        assert ev.status == "fail"
        assert ev.detail == ""


class TestProofCategory:
    def test_all_pass(self):
        cat = ProofCategory(name="test", description="desc")
        cat.evidence = [
            EvidenceItem("a", "pass"),
            EvidenceItem("b", "pass"),
        ]
        cat.compute_status()
        assert cat.status == "pass"

    def test_any_fail(self):
        cat = ProofCategory(name="test", description="desc")
        cat.evidence = [
            EvidenceItem("a", "pass"),
            EvidenceItem("b", "fail"),
        ]
        cat.compute_status()
        assert cat.status == "fail"

    def test_all_warn(self):
        cat = ProofCategory(name="test", description="desc")
        cat.evidence = [
            EvidenceItem("a", "warn"),
            EvidenceItem("b", "warn"),
        ]
        cat.compute_status()
        assert cat.status == "partial"

    def test_empty_evidence(self):
        cat = ProofCategory(name="test", description="desc")
        cat.compute_status()
        assert cat.status == "fail"

    def test_mixed_pass_warn(self):
        cat = ProofCategory(name="test", description="desc")
        cat.evidence = [
            EvidenceItem("a", "pass"),
            EvidenceItem("b", "warn"),
        ]
        cat.compute_status()
        assert cat.status == "partial"


class TestProofReport:
    def test_all_pass_verdict(self):
        report = ProofReport(timestamp="t", version="v")
        cat1 = ProofCategory(name="c1", description="d1")
        cat1.evidence = [EvidenceItem("a", "pass")]
        cat2 = ProofCategory(name="c2", description="d2")
        cat2.evidence = [EvidenceItem("b", "pass")]
        report.categories = [cat1, cat2]
        report.compute_verdict()
        assert report.verdict == "pass"

    def test_any_fail_verdict(self):
        report = ProofReport(timestamp="t", version="v")
        cat1 = ProofCategory(name="c1", description="d1")
        cat1.evidence = [EvidenceItem("a", "pass")]
        cat2 = ProofCategory(name="c2", description="d2")
        cat2.evidence = [EvidenceItem("b", "fail")]
        report.categories = [cat1, cat2]
        report.compute_verdict()
        assert report.verdict == "fail"

    def test_partial_verdict(self):
        report = ProofReport(timestamp="t", version="v")
        cat1 = ProofCategory(name="c1", description="d1")
        cat1.evidence = [EvidenceItem("a", "pass")]
        cat2 = ProofCategory(name="c2", description="d2")
        cat2.evidence = [EvidenceItem("b", "warn")]
        report.categories = [cat1, cat2]
        report.compute_verdict()
        assert report.verdict == "partial"


# ---------------------------------------------------------------------------
# Failover readiness tests
# ---------------------------------------------------------------------------


class TestFailoverReadiness:
    def test_all_present(self, tmp_project: Path):
        cat = check_failover_readiness(tmp_project)
        cat.compute_status()
        assert cat.status == "pass"
        assert len(cat.evidence) == 5

    def test_missing_drill_script(self, tmp_project: Path):
        (tmp_project / "scripts" / "failover_drill.py").unlink()
        cat = check_failover_readiness(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"
        assert any(e.status == "fail" and "drill" in e.description.lower()
                    for e in cat.evidence)

    def test_missing_runbook(self, tmp_project: Path):
        (tmp_project / "docs" / "FAILOVER-RUNBOOK.md").unlink()
        cat = check_failover_readiness(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"
        assert any(e.status == "fail" and "runbook" in e.description.lower()
                    for e in cat.evidence)

    def test_drill_without_drillstep_class(self, tmp_project: Path):
        (tmp_project / "scripts" / "failover_drill.py").write_text(
            "# empty drill\n", encoding="utf-8"
        )
        cat = check_failover_readiness(tmp_project)
        assert any(e.status == "warn" and "DrillStep" in e.description
                    for e in cat.evidence)

    def test_runbook_without_postgres_section(self, tmp_project: Path):
        (tmp_project / "docs" / "FAILOVER-RUNBOOK.md").write_text(
            "# Generic runbook\n", encoding="utf-8"
        )
        cat = check_failover_readiness(tmp_project)
        assert any(e.status == "warn" and "runbook" in e.description.lower()
                    for e in cat.evidence)

    def test_missing_sentinel(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "redis-sentinel-statefulset.yaml").unlink()
        cat = check_failover_readiness(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_backup_cronjob(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "postgres-backup-cronjob.yaml").unlink()
        cat = check_failover_readiness(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"


# ---------------------------------------------------------------------------
# Autoscaling config tests
# ---------------------------------------------------------------------------


class TestAutoscalingConfig:
    def test_all_present(self, tmp_project: Path):
        cat = check_autoscaling_config(tmp_project)
        cat.compute_status()
        assert cat.status == "pass"
        assert len(cat.evidence) >= 7  # 3 scalers + 3 deployments + 1 PDB

    def test_missing_gpu_scaler(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "keda-gpu-scaler.yaml").unlink()
        cat = check_autoscaling_config(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_scaler_missing_triggers(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "keda-gpu-scaler.yaml").write_text(
            "apiVersion: keda.sh/v1alpha1\nspec:\n  minReplicaCount: 1\n"
            "  maxReplicaCount: 5\n",
            encoding="utf-8",
        )
        cat = check_autoscaling_config(tmp_project)
        assert any(e.status == "fail" and "triggers" in e.description
                    for e in cat.evidence)

    def test_scaler_missing_min_replica(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "keda-cpu-scaler.yaml").write_text(
            "apiVersion: keda.sh/v1alpha1\nspec:\n  maxReplicaCount: 5\n  triggers:\n"
            "    - type: rabbitmq\n",
            encoding="utf-8",
        )
        cat = check_autoscaling_config(tmp_project)
        assert any(e.status == "fail" and "minReplicaCount" in e.description
                    for e in cat.evidence)

    def test_missing_gpu_deployment(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "gpu-worker-deployment.yaml").unlink()
        cat = check_autoscaling_config(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_deployment_without_resources(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "gpu-worker-deployment.yaml").write_text(
            "kind: Deployment\nspec:\n  containers: []\n",
            encoding="utf-8",
        )
        cat = check_autoscaling_config(tmp_project)
        assert any(e.status == "fail" and "GPU" in e.description
                    and "missing resource" in e.description
                    for e in cat.evidence)

    def test_missing_pdb(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates" / "pdb.yaml").unlink()
        cat = check_autoscaling_config(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_pdb_without_max_unavailable(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates" / "pdb.yaml").write_text(
            "kind: PodDisruptionBudget\nspec:\n  minAvailable: 1\n",
            encoding="utf-8",
        )
        cat = check_autoscaling_config(tmp_project)
        assert any(e.status == "warn" and "PDB" in e.description
                    for e in cat.evidence)


# ---------------------------------------------------------------------------
# Crash recovery tests
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_all_present(self, tmp_project: Path):
        cat = check_crash_recovery(tmp_project)
        cat.compute_status()
        assert cat.status == "pass"
        assert len(cat.evidence) == 5

    def test_missing_pipeline(self, tmp_project: Path):
        (tmp_project / "ocr_gpu_async.py").unlink()
        cat = check_crash_recovery(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"
        assert any(e.status == "fail" and "pipeline" in e.description.lower()
                    for e in cat.evidence)

    def test_pipeline_without_temp_folder(self, tmp_project: Path):
        (tmp_project / "ocr_gpu_async.py").write_text(
            "# no temp handling\n", encoding="utf-8"
        )
        cat = check_crash_recovery(tmp_project)
        assert any(e.status == "fail" and "resume" in e.description.lower()
                    for e in cat.evidence)

    def test_missing_cleanup_command(self, tmp_project: Path):
        (tmp_project / "coordinator" / "jobs" / "management" / "commands"
         / "cleanup_old_jobs.py").unlink()
        cat = check_crash_recovery(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_purge_command(self, tmp_project: Path):
        (tmp_project / "coordinator" / "jobs" / "management" / "commands"
         / "purge_temp_files.py").unlink()
        cat = check_crash_recovery(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_errback_handling(self, tmp_project: Path):
        (tmp_project / "coordinator" / "jobs" / "tasks.py").write_text(
            "# no error handling\n", encoding="utf-8"
        )
        cat = check_crash_recovery(tmp_project)
        assert any(e.status == "fail" and "errback" in e.description.lower()
                    for e in cat.evidence)

    def test_scale_test_without_crash_mode(self, tmp_project: Path):
        (tmp_project / "scale_test.py").write_text(
            "# just testing\n", encoding="utf-8"
        )
        cat = check_crash_recovery(tmp_project)
        assert any(e.status == "warn" and "crash recovery" in e.description.lower()
                    for e in cat.evidence)

    def test_missing_scale_test(self, tmp_project: Path):
        (tmp_project / "scale_test.py").unlink()
        cat = check_crash_recovery(tmp_project)
        assert any(e.status == "fail" and "scale test" in e.description.lower()
                    for e in cat.evidence)


# ---------------------------------------------------------------------------
# Infrastructure HA tests
# ---------------------------------------------------------------------------


class TestInfrastructureHA:
    def test_all_present(self, tmp_project: Path):
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "pass"
        assert len(cat.evidence) == 6

    def test_missing_ha_compose(self, tmp_project: Path):
        (tmp_project / "coordinator" / "docker-compose.ha.yml").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_prometheus_rule(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "prometheusrule.yaml").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_prometheus_rule_without_alerts(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "prometheusrule.yaml").write_text(
            "kind: PrometheusRule\nspec: {}\n", encoding="utf-8"
        )
        cat = check_infrastructure_ha(tmp_project)
        assert any(e.status == "warn" and "PrometheusRule" in e.description
                    for e in cat.evidence)

    def test_missing_grafana(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "grafana-dashboard-configmap.yaml").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_health_router(self, tmp_project: Path):
        (tmp_project / "api" / "routers" / "health.py").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_health_router_without_endpoint(self, tmp_project: Path):
        (tmp_project / "api" / "routers" / "health.py").write_text(
            "# empty router\n", encoding="utf-8"
        )
        cat = check_infrastructure_ha(tmp_project)
        assert any(e.status == "warn" and "Health" in e.description
                    for e in cat.evidence)

    def test_missing_healthcheck_sh(self, tmp_project: Path):
        (tmp_project / "healthcheck.sh").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_missing_service_monitor(self, tmp_project: Path):
        (tmp_project / "helm" / "ocr-local" / "templates"
         / "servicemonitor.yaml").unlink()
        cat = check_infrastructure_ha(tmp_project)
        cat.compute_status()
        assert cat.status == "fail"

    def test_ha_compose_without_cluster(self, tmp_project: Path):
        (tmp_project / "coordinator" / "docker-compose.ha.yml").write_text(
            "services:\n  redis:\n    image: redis\n", encoding="utf-8"
        )
        cat = check_infrastructure_ha(tmp_project)
        # Still has redis reference so sentinel check passes, but rabbitmq cluster missing
        assert any(e.status in ("pass", "warn") and "HA compose" in e.description
                    for e in cat.evidence)


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_contains_verdict(self, tmp_project: Path):
        report = build_report(tmp_project)
        text = format_text(report)
        assert "VERDICT:" in text
        assert "PASS" in text

    def test_contains_categories(self, tmp_project: Path):
        report = build_report(tmp_project)
        text = format_text(report)
        assert "failover_readiness" in text
        assert "autoscaling_config" in text
        assert "crash_recovery" in text
        assert "infrastructure_ha" in text

    def test_contains_evidence_markers(self, tmp_project: Path):
        report = build_report(tmp_project)
        text = format_text(report)
        assert "[PASS]" in text


class TestFormatJson:
    def test_valid_json(self, tmp_project: Path):
        report = build_report(tmp_project)
        text = format_json(report)
        parsed = json.loads(text)
        assert "verdict" in parsed
        assert "categories" in parsed
        assert len(parsed["categories"]) == 4

    def test_json_has_evidence(self, tmp_project: Path):
        report = build_report(tmp_project)
        parsed = json.loads(format_json(report))
        for cat in parsed["categories"]:
            assert "evidence" in cat
            assert len(cat["evidence"]) > 0

    def test_json_verdict_pass(self, tmp_project: Path):
        report = build_report(tmp_project)
        parsed = json.loads(format_json(report))
        assert parsed["verdict"] == "pass"


class TestFormatMarkdown:
    def test_contains_header(self, tmp_project: Path):
        report = build_report(tmp_project)
        md = format_markdown(report)
        assert "# HA / Autoscaling / Crash-Recovery Proof Report" in md

    def test_contains_table(self, tmp_project: Path):
        report = build_report(tmp_project)
        md = format_markdown(report)
        assert "| Status | Check | Detail |" in md

    def test_contains_score(self, tmp_project: Path):
        report = build_report(tmp_project)
        md = format_markdown(report)
        assert "**Score**:" in md


# ---------------------------------------------------------------------------
# Integration tests against real project root
# ---------------------------------------------------------------------------


class TestRealProject:
    def test_build_report_real_root(self, real_root: Path):
        """Run against the actual project to verify all checks pass."""
        report = build_report(real_root)
        assert report.verdict == "pass", (
            f"Expected pass but got {report.verdict}. "
            + "; ".join(
                f"{c.name}: {e.description} [{e.status}]"
                for c in report.categories
                for e in c.evidence
                if e.status != "pass"
            )
        )

    def test_all_categories_present(self, real_root: Path):
        report = build_report(real_root)
        names = {c.name for c in report.categories}
        assert names == {
            "failover_readiness",
            "autoscaling_config",
            "crash_recovery",
            "infrastructure_ha",
        }


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_exit_0_on_pass(self, tmp_project: Path):
        rc = main(["--project-root", str(tmp_project)])
        assert rc == 0

    def test_exit_1_on_fail(self, tmp_project: Path):
        (tmp_project / "scripts" / "failover_drill.py").unlink()
        rc = main(["--project-root", str(tmp_project)])
        assert rc == 1

    def test_json_output(self, tmp_project: Path, capsys):
        main(["--project-root", str(tmp_project), "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "verdict" in parsed

    def test_report_output(self, tmp_project: Path, tmp_path: Path):
        report_file = tmp_path / "report.md"
        main(["--project-root", str(tmp_project), "--report", str(report_file)])
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "# HA / Autoscaling / Crash-Recovery Proof Report" in content

    def test_default_project_root(self, capsys, real_root: Path):
        """CLI auto-detects project root when --project-root is not given."""
        rc = main([])
        assert rc == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_project(self, tmp_path: Path):
        """All checks fail on a completely empty directory."""
        report = build_report(tmp_path)
        assert report.verdict == "fail"
        for cat in report.categories:
            assert cat.status == "fail"

    def test_partial_project(self, tmp_path: Path):
        """Some checks pass, some fail."""
        # Only create the healthcheck script
        (tmp_path / "healthcheck.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        report = build_report(tmp_path)
        assert report.verdict == "fail"
        # Infrastructure HA should have at least one pass
        infra_cat = next(c for c in report.categories if c.name == "infrastructure_ha")
        assert any(e.status == "pass" for e in infra_cat.evidence)
        assert any(e.status == "fail" for e in infra_cat.evidence)

    def test_unreadable_file(self, tmp_project: Path):
        """Binary/corrupt files do not crash the framework."""
        (tmp_project / "scripts" / "failover_drill.py").write_bytes(
            b"\x00\x01\x02\xff\xfe"
        )
        cat = check_failover_readiness(tmp_project)
        # Should not raise; may produce warn for missing class
        assert cat is not None
