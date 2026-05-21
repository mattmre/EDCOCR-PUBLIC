"""Tests for Phase 7D multi-replica cluster validation.

Validates that all HA infrastructure components work together correctly:
- Helm template rendering with all HA features enabled simultaneously
- Configuration consistency across rendered resources
- Failover resilience at the Django model layer

Requires: helm CLI available on PATH (for Helm rendering tests).

Run with: cd coordinator && python -m pytest jobs/tests/test_cluster_validation.py -v
"""

import os
import shutil
import subprocess
import unittest
from datetime import timedelta

import yaml
from django.test import TestCase
from django.utils import timezone

from jobs.models import Job, Worker

# ---------------------------------------------------------------------------
# Helm chart path and common flags
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
CHART_PATH = os.path.join(_PROJECT_ROOT, "helm", "ocr-local")

_REQUIRED_SETS = [
    "--set", "secrets.djangoSecretKey=test",
    "--set", "secrets.postgresPassword=test",
    "--set", "secrets.rabbitmqPassword=test",
    "--set", "secrets.redisPassword=test",
    "--set", "secrets.flowerPassword=test-flower-password",
]

_HA_SETS = [
    "--set", "rabbitmq.replicas=3",
    "--set", "redis.sentinel.enabled=true",
    "--set", "redis.sentinel.quorum=3",
    "--set", "postgresql.backup.enabled=true",
    "--set", "prometheus.enabled=true",
    "--set", "prometheus.serviceMonitor.enabled=true",
    "--set", "prometheus.rules.enabled=true",
    "--set", "prometheus.grafana.enabled=true",
    "--set", "secrets.metricsApiKey=test-key",
    "--set", "coordinator.replicas=2",
    "--set", "gpuWorker.replicas=4",
]

HELM_AVAILABLE = shutil.which("helm") is not None


def _helm_template_ha():
    """Run helm template with all HA flags and return parsed YAML documents."""
    cmd = [
        "helm", "template", "ha-test", CHART_PATH,
    ] + list(_REQUIRED_SETS) + list(_HA_SETS)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    docs = []
    for doc in yaml.safe_load_all(result.stdout):
        if doc is not None:
            docs.append(doc)
    return docs


def _find_resources(docs, kind, name_contains=None):
    """Filter rendered documents by kind and optional name substring."""
    results = []
    for doc in docs:
        if doc.get("kind") == kind:
            if name_contains is None:
                results.append(doc)
            elif name_contains in doc.get("metadata", {}).get("name", ""):
                results.append(doc)
    return results


# ============================================================================
# TestMultiReplicaHelmRendering
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestMultiReplicaHelmRendering(unittest.TestCase):
    """Validate the complete HA Helm deployment renders correctly
    when all HA features are enabled simultaneously."""

    @classmethod
    def setUpClass(cls):
        """Render the full HA chart once and cache for all tests."""
        super().setUpClass()
        cls.docs = _helm_template_ha()

    # ------------------------------------------------------------------
    # 1. Full HA renders without errors
    # ------------------------------------------------------------------
    def test_full_ha_renders_without_errors(self):
        """All HA flags enabled, helm template exits 0 and produces docs."""
        self.assertGreater(
            len(self.docs), 0,
            "helm template with all HA flags must produce at least one resource"
        )

    # ------------------------------------------------------------------
    # 2. Expected resource kinds present
    # ------------------------------------------------------------------
    def test_expected_resource_kinds_present(self):
        """All expected Kubernetes resource kinds should be rendered."""
        rendered_kinds = {doc["kind"] for doc in self.docs if doc}
        expected_kinds = {
            "Deployment", "StatefulSet", "Service", "Secret",
            "ConfigMap", "PodDisruptionBudget",
        }
        missing = expected_kinds - rendered_kinds
        self.assertEqual(
            missing, set(),
            f"Expected resource kinds missing from rendered chart: {missing}"
        )

    # ------------------------------------------------------------------
    # 3. RabbitMQ replicas
    # ------------------------------------------------------------------
    def test_rabbitmq_replicas_set(self):
        """RabbitMQ StatefulSet has replicas=3."""
        sts_list = _find_resources(self.docs, "StatefulSet", "rabbitmq")
        # Filter to the main RabbitMQ StatefulSet (not sentinel-related)
        rabbitmq_sts = [
            s for s in sts_list
            if "sentinel" not in s["metadata"]["name"]
            and "replica" not in s["metadata"]["name"]
        ]
        self.assertEqual(
            len(rabbitmq_sts), 1,
            f"Expected exactly 1 rabbitmq StatefulSet, found {len(rabbitmq_sts)}"
        )
        self.assertEqual(
            rabbitmq_sts[0]["spec"]["replicas"], 3,
            "RabbitMQ StatefulSet should have replicas=3"
        )

    # ------------------------------------------------------------------
    # 4. Redis Sentinel StatefulSet exists with replicas=3
    # ------------------------------------------------------------------
    def test_redis_sentinel_created(self):
        """Redis Sentinel StatefulSet exists with replicas=3."""
        sentinel_sts = _find_resources(self.docs, "StatefulSet", "redis-sentinel")
        self.assertGreaterEqual(
            len(sentinel_sts), 1,
            "At least one redis-sentinel StatefulSet should be rendered"
        )
        self.assertEqual(
            sentinel_sts[0]["spec"]["replicas"], 3,
            "Redis Sentinel StatefulSet should have replicas=3 (sentinelReplicas default)"
        )

    # ------------------------------------------------------------------
    # 5. Redis Replica StatefulSet exists
    # ------------------------------------------------------------------
    def test_redis_replica_created(self):
        """Redis Replica StatefulSet exists."""
        replica_sts = _find_resources(self.docs, "StatefulSet", "redis-replica")
        self.assertEqual(
            len(replica_sts), 1,
            "Exactly one redis-replica StatefulSet should be rendered"
        )

    # ------------------------------------------------------------------
    # 6. Postgres backup CronJob exists
    # ------------------------------------------------------------------
    def test_postgres_backup_cronjob_created(self):
        """CronJob for postgres backup exists."""
        cronjobs = _find_resources(self.docs, "CronJob", "postgres-backup")
        self.assertEqual(
            len(cronjobs), 1,
            "Exactly one postgres-backup CronJob should be rendered"
        )
        self.assertEqual(cronjobs[0]["apiVersion"], "batch/v1")

    # ------------------------------------------------------------------
    # 7. Prometheus monitoring stack complete
    # ------------------------------------------------------------------
    def test_prometheus_monitoring_stack_complete(self):
        """ServiceMonitor + PrometheusRule + Grafana ConfigMap all present."""
        service_monitors = _find_resources(self.docs, "ServiceMonitor")
        self.assertGreaterEqual(
            len(service_monitors), 1,
            "At least one ServiceMonitor should be rendered"
        )

        prom_rules = _find_resources(self.docs, "PrometheusRule")
        self.assertGreaterEqual(
            len(prom_rules), 1,
            "At least one PrometheusRule should be rendered"
        )

        grafana_cms = _find_resources(self.docs, "ConfigMap", "grafana-dashboard")
        self.assertGreaterEqual(
            len(grafana_cms), 1,
            "At least one Grafana dashboard ConfigMap should be rendered"
        )

    # ------------------------------------------------------------------
    # 8. Coordinator replicas
    # ------------------------------------------------------------------
    def test_coordinator_replicas(self):
        """Coordinator Deployment has replicas=2."""
        deployments = _find_resources(self.docs, "Deployment", "coordinator")
        # Filter to the main coordinator Deployment (not celery-coordinator)
        coord_deps = [
            d for d in deployments
            if d["metadata"]["name"] == "ha-test-coordinator"
        ]
        self.assertEqual(
            len(coord_deps), 1,
            f"Expected exactly 1 coordinator Deployment named 'ha-test-coordinator', "
            f"found: {[d['metadata']['name'] for d in deployments]}"
        )
        self.assertEqual(
            coord_deps[0]["spec"]["replicas"], 2,
            "Coordinator Deployment should have replicas=2"
        )

    # ------------------------------------------------------------------
    # 9. GPU Worker replicas
    # ------------------------------------------------------------------
    def test_gpu_worker_replicas(self):
        """GPU Worker Deployment has replicas=4."""
        gpu_deps = _find_resources(self.docs, "Deployment", "gpu-worker")
        self.assertEqual(
            len(gpu_deps), 1,
            "Exactly one gpu-worker Deployment should be rendered"
        )
        self.assertEqual(
            gpu_deps[0]["spec"]["replicas"], 4,
            "GPU Worker Deployment should have replicas=4"
        )

    # ------------------------------------------------------------------
    # 10. All 5 PDBs present
    # ------------------------------------------------------------------
    def test_all_pdbs_present(self):
        """All 5 PDBs present: gpu-worker, coordinator, postgres, rabbitmq, redis."""
        pdbs = _find_resources(self.docs, "PodDisruptionBudget")
        pdb_names = {p["metadata"]["name"] for p in pdbs}

        expected_substrings = ["gpu-worker", "coordinator", "postgres", "rabbitmq", "redis"]
        for substr in expected_substrings:
            matches = [n for n in pdb_names if substr in n]
            self.assertGreaterEqual(
                len(matches), 1,
                f"Expected a PDB containing '{substr}' in its name. "
                f"Found PDBs: {sorted(pdb_names)}"
            )

        self.assertGreaterEqual(
            len(pdbs), 5,
            f"Expected at least 5 PDBs, got {len(pdbs)}: {sorted(pdb_names)}"
        )

    # ------------------------------------------------------------------
    # 11. Services reference correct selectors
    # ------------------------------------------------------------------
    def test_services_reference_correct_selectors(self):
        """All Services have matching selector labels that include
        the common selectorLabels (name + instance)."""
        services = _find_resources(self.docs, "Service")
        self.assertGreater(
            len(services), 0,
            "At least one Service should be rendered"
        )
        for svc in services:
            svc_name = svc["metadata"]["name"]
            selector = svc.get("spec", {}).get("selector", {})
            self.assertIn(
                "app.kubernetes.io/name", selector,
                f"Service '{svc_name}' selector must include app.kubernetes.io/name"
            )
            self.assertIn(
                "app.kubernetes.io/instance", selector,
                f"Service '{svc_name}' selector must include app.kubernetes.io/instance"
            )
            # The instance label should match our release name
            self.assertEqual(
                selector["app.kubernetes.io/instance"], "ha-test",
                f"Service '{svc_name}' selector instance must be 'ha-test'"
            )


# ============================================================================
# TestConfigurationConsistency
# ============================================================================
@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not available")
class TestConfigurationConsistency(unittest.TestCase):
    """Validate configuration values are consistent across components."""

    @classmethod
    def setUpClass(cls):
        """Render the full HA chart once and cache for all tests."""
        super().setUpClass()
        cls.docs = _helm_template_ha()

    # ------------------------------------------------------------------
    # 12. Secret contains all required keys
    # ------------------------------------------------------------------
    def test_secret_contains_all_required_keys(self):
        """The rendered Secret contains all required keys."""
        secrets = _find_resources(self.docs, "Secret")
        self.assertGreaterEqual(
            len(secrets), 1,
            "At least one Secret should be rendered"
        )

        # Find the main secret (ha-test-secret)
        main_secret = None
        for s in secrets:
            if s["metadata"]["name"] == "ha-test-secret":
                main_secret = s
                break
        self.assertIsNotNone(
            main_secret,
            f"Secret 'ha-test-secret' not found. Secrets: "
            f"{[s['metadata']['name'] for s in secrets]}"
        )

        required_keys = [
            "DJANGO_SECRET_KEY",
            "POSTGRES_PASSWORD",
            "DATABASE_URL",
            "RABBITMQ_USER",
            "RABBITMQ_PASSWORD",
            "CELERY_BROKER_URL",
            "REDIS_PASSWORD",
            "REDIS_URL",
            "CELERY_RESULT_BACKEND",
            "FLOWER_PASSWORD",
            "METRICS_API_KEY",
        ]
        data_keys = set(main_secret.get("data", {}).keys())
        for key in required_keys:
            self.assertIn(
                key, data_keys,
                f"Secret must contain key '{key}'. Found keys: {sorted(data_keys)}"
            )

    # ------------------------------------------------------------------
    # 13. Coordinator env references correct secret
    # ------------------------------------------------------------------
    def test_coordinator_env_references_secret(self):
        """Coordinator Deployment env vars reference the correct secret name."""
        deployments = _find_resources(self.docs, "Deployment")
        coord_dep = None
        for d in deployments:
            if d["metadata"]["name"] == "ha-test-coordinator":
                coord_dep = d
                break
        self.assertIsNotNone(
            coord_dep,
            "Coordinator Deployment 'ha-test-coordinator' not found"
        )

        # Check containers for secretRef
        containers = coord_dep["spec"]["template"]["spec"]["containers"]
        found_secret_ref = False
        for container in containers:
            env_from = container.get("envFrom", [])
            for ef in env_from:
                secret_ref = ef.get("secretRef", {})
                if secret_ref.get("name") == "ha-test-secret":
                    found_secret_ref = True
                    break

        # Also check initContainers
        init_containers = coord_dep["spec"]["template"]["spec"].get("initContainers", [])
        for container in init_containers:
            env_from = container.get("envFrom", [])
            for ef in env_from:
                secret_ref = ef.get("secretRef", {})
                if secret_ref.get("name") == "ha-test-secret":
                    found_secret_ref = True
                    break

        self.assertTrue(
            found_secret_ref,
            "Coordinator Deployment must reference 'ha-test-secret' via envFrom.secretRef"
        )

    def test_coordinator_probes_use_metrics_api_key_when_configured(self):
        """Coordinator probes should use the configured metrics API key."""
        deployments = _find_resources(self.docs, "Deployment")
        coord_dep = next(
            (d for d in deployments if d["metadata"]["name"] == "ha-test-coordinator"),
            None,
        )
        self.assertIsNotNone(coord_dep)

        container = coord_dep["spec"]["template"]["spec"]["containers"][0]
        for probe_name in ("livenessProbe", "readinessProbe", "startupProbe"):
            probe = container[probe_name]["httpGet"]
            self.assertEqual(
                probe["httpHeaders"],
                [{"name": "X-Api-Key", "value": "test-key"}],
                f"{probe_name} must use the configured metrics API key",
            )

    # ------------------------------------------------------------------
    # 14. Celery workers share same secret
    # ------------------------------------------------------------------
    def test_celery_workers_share_same_secret(self):
        """Celery coordinator, GPU worker, and CPU worker all reference the same secret."""
        deployments = _find_resources(self.docs, "Deployment")

        expected_components = {
            "ha-test-celery-coordinator": False,
            "ha-test-gpu-worker": False,
            "ha-test-cpu-worker": False,
        }

        for dep in deployments:
            dep_name = dep["metadata"]["name"]
            if dep_name not in expected_components:
                continue

            containers = dep["spec"]["template"]["spec"]["containers"]
            for container in containers:
                env_from = container.get("envFrom", [])
                for ef in env_from:
                    secret_ref = ef.get("secretRef", {})
                    if secret_ref.get("name") == "ha-test-secret":
                        expected_components[dep_name] = True
                        break

        for dep_name, found in expected_components.items():
            self.assertTrue(
                found,
                f"Deployment '{dep_name}' must reference 'ha-test-secret' via "
                f"envFrom.secretRef"
            )


# ============================================================================
# TestFailoverResilience (Django model-level, no Helm)
# ============================================================================
class TestFailoverResilience(TestCase):
    """Validate the coordinator handles failure scenarios correctly
    at the Django model/query layer."""

    # ------------------------------------------------------------------
    # 15. Offline worker not counted in GPU available
    # ------------------------------------------------------------------
    def test_worker_offline_not_counted_in_gpu_available(self):
        """Create Worker(status='offline', gpu_available=True), verify
        metrics exclude it from GPU available count."""
        Worker.objects.create(
            hostname="gpu-worker-offline-1",
            status=Worker.Status.OFFLINE,
            gpu_available=True,
            gpu_model="NVIDIA A100",
            gpu_vram_mb=40960,
        )
        Worker.objects.create(
            hostname="gpu-worker-online-1",
            status=Worker.Status.ONLINE,
            gpu_available=True,
            gpu_model="NVIDIA A100",
            gpu_vram_mb=40960,
        )

        # Replicate the metrics query from views.py
        gpu_available_count = Worker.objects.filter(
            gpu_available=True,
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
        ).count()

        self.assertEqual(
            gpu_available_count, 1,
            "Only online/busy workers with gpu_available=True should be counted. "
            "Offline workers must be excluded."
        )

    # ------------------------------------------------------------------
    # 16. Multiple workers with different GPU indices
    # ------------------------------------------------------------------
    def test_multiple_workers_different_gpus(self):
        """Create workers with different gpu_index values, verify all are tracked."""
        for i in range(4):
            Worker.objects.create(
                hostname=f"gpu-node-{i}",
                status=Worker.Status.ONLINE,
                gpu_available=True,
                gpu_model="NVIDIA A100",
                gpu_vram_mb=40960,
                gpu_index=i,
            )

        workers = Worker.objects.filter(gpu_available=True)
        self.assertEqual(workers.count(), 4)

        gpu_indices = set(workers.values_list("gpu_index", flat=True))
        self.assertEqual(gpu_indices, {0, 1, 2, 3})

    # ------------------------------------------------------------------
    # 17. Stale heartbeat detection
    # ------------------------------------------------------------------
    def test_stale_heartbeat_detection(self):
        """Create worker with last_heartbeat 10 minutes ago, verify
        it can be detected via query."""
        stale_time = timezone.now() - timedelta(minutes=10)
        fresh_time = timezone.now() - timedelta(seconds=30)

        Worker.objects.create(
            hostname="stale-worker-1",
            status=Worker.Status.ONLINE,
            gpu_available=True,
            last_heartbeat=stale_time,
        )
        Worker.objects.create(
            hostname="fresh-worker-1",
            status=Worker.Status.ONLINE,
            gpu_available=True,
            last_heartbeat=fresh_time,
        )

        # Detect stale workers: heartbeat older than 5 minutes
        stale_threshold = timezone.now() - timedelta(minutes=5)
        stale_workers = Worker.objects.filter(
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
            last_heartbeat__lt=stale_threshold,
        )

        self.assertEqual(stale_workers.count(), 1)
        self.assertEqual(stale_workers.first().hostname, "stale-worker-1")

        # Fresh workers should not appear in stale query
        fresh_workers = Worker.objects.filter(
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY],
            last_heartbeat__gte=stale_threshold,
        )
        self.assertEqual(fresh_workers.count(), 1)
        self.assertEqual(fresh_workers.first().hostname, "fresh-worker-1")

    # ------------------------------------------------------------------
    # 18. Job survives worker failure
    # ------------------------------------------------------------------
    def test_job_survives_worker_failure(self):
        """Create a job in 'processing' state, mark its worker offline,
        verify job status is still 'processing' (coordinator handles
        retry via chord errback, not immediate status change)."""
        worker = Worker.objects.create(
            hostname="gpu-worker-doomed",
            status=Worker.Status.ONLINE,
            gpu_available=True,
        )

        job = Job.objects.create(
            source_file="/nfs/jobs/test-doc.pdf",
            status=Job.Status.PROCESSING,
            assigned_worker=worker.hostname,
            total_pages=10,
            pages_completed=3,
        )

        # Simulate worker failure: mark offline
        worker.status = Worker.Status.OFFLINE
        worker.save()

        # Reload job from DB
        job.refresh_from_db()

        # Job status must remain 'processing' -- the coordinator handles
        # retry/reassignment via Celery chord errback, not by changing
        # job status when a worker goes offline.
        self.assertEqual(
            job.status, Job.Status.PROCESSING,
            "Job status must remain 'processing' when its worker goes offline. "
            "The coordinator handles retry via chord errback."
        )
        self.assertEqual(job.assigned_worker, "gpu-worker-doomed")
        self.assertEqual(job.pages_completed, 3)

        # Verify the worker is indeed offline
        worker.refresh_from_db()
        self.assertEqual(worker.status, Worker.Status.OFFLINE)
