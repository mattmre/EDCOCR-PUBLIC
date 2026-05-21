"""Tests for Phase 7D HA completion: PostgreSQL backups and Redis Sentinel.

Covers:
- Helm template rendering for postgres-backup-cronjob.yaml
- Helm template rendering for redis-sentinel-statefulset.yaml
- Redis Sentinel configuration in coordinator/coordinator/settings.py

Requires: helm CLI available on PATH.

Run with: cd coordinator && python -m pytest jobs/tests/test_ha_completion.py -v
"""

import importlib
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

import yaml
from django.test import TestCase

# Compute chart path from project root
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
CHART_PATH = os.path.join(_PROJECT_ROOT, "helm", "ocr-local")

# Required --set flags for chart secrets (all templates require these)
_REQUIRED_SETS = [
    "--set", "secrets.djangoSecretKey=test",
    "--set", "secrets.postgresPassword=test",
    "--set", "secrets.rabbitmqPassword=test",
    "--set", "secrets.redisPassword=test",
    "--set", "secrets.flowerPassword=test-flower-password",
]

# Check if helm is available
HELM_AVAILABLE = shutil.which("helm") is not None


def _helm_template(*extra_sets):
    """Run helm template and return parsed YAML documents as a list of dicts.

    Each document in the multi-document YAML output is returned as a dict.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    cmd = [
        "helm", "template", "test-release", CHART_PATH,
    ] + list(_REQUIRED_SETS) + list(extra_sets)
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


@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not found on PATH")
class TestHelmPostgresBackup(unittest.TestCase):
    """Tests for the PostgreSQL backup CronJob Helm template."""

    def test_backup_cronjob_disabled_by_default(self):
        """With default values, no CronJob resource should be rendered."""
        docs = _helm_template()
        cronjobs = _find_resources(docs, "CronJob")
        self.assertEqual(
            len(cronjobs), 0,
            "No CronJob should be rendered when postgresql.backup.enabled is false (default)"
        )

    def test_backup_cronjob_enabled(self):
        """When postgresql.backup.enabled=true, a CronJob is rendered."""
        docs = _helm_template("--set", "postgresql.backup.enabled=true")
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        self.assertEqual(
            len(cronjobs), 1,
            "Exactly one postgres-backup CronJob should be rendered"
        )
        cj = cronjobs[0]
        self.assertEqual(cj["metadata"]["name"], "test-release-postgres-backup")
        self.assertEqual(cj["apiVersion"], "batch/v1")

    def test_backup_schedule_configurable(self):
        """A custom schedule value appears in the rendered CronJob spec."""
        custom_schedule = "30 3 * * 0"
        docs = _helm_template(
            "--set", "postgresql.backup.enabled=true",
            "--set", f"postgresql.backup.schedule={custom_schedule}",
        )
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        self.assertEqual(len(cronjobs), 1)
        self.assertEqual(
            cronjobs[0]["spec"]["schedule"], custom_schedule,
            f"CronJob schedule should be '{custom_schedule}'"
        )

    def test_backup_pvc_created(self):
        """When backup is enabled, a PersistentVolumeClaim is created with correct storage."""
        docs = _helm_template(
            "--set", "postgresql.backup.enabled=true",
            "--set", "postgresql.backup.storage=50Gi",
        )
        pvcs = _find_resources(docs, "PersistentVolumeClaim", "postgres-backup")
        self.assertEqual(
            len(pvcs), 1,
            "Exactly one postgres-backup PVC should be rendered"
        )
        pvc = pvcs[0]
        self.assertEqual(pvc["metadata"]["name"], "test-release-postgres-backup")
        storage = pvc["spec"]["resources"]["requests"]["storage"]
        self.assertEqual(storage, "50Gi")

    def test_backup_pvc_not_created_when_disabled(self):
        """When backup is disabled (default), no backup PVC is rendered."""
        docs = _helm_template()
        pvcs = _find_resources(docs, "PersistentVolumeClaim", "postgres-backup")
        self.assertEqual(
            len(pvcs), 0,
            "No postgres-backup PVC should be rendered when backup is disabled"
        )

    def test_backup_retention_in_script(self):
        """The retention count value appears in the CronJob container script."""
        docs = _helm_template(
            "--set", "postgresql.backup.enabled=true",
            "--set", "postgresql.backup.retentionCount=14",
        )
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        self.assertEqual(len(cronjobs), 1)
        containers = (
            cronjobs[0]["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"]
        )
        self.assertEqual(len(containers), 1)
        # The command is a list: ["sh", "-c", "<script>"]
        script = containers[0]["command"][2]
        self.assertIn(
            "RETENTION=14", script,
            "Retention count of 14 should appear in the backup script"
        )

    def test_backup_default_schedule(self):
        """Default schedule from values.yaml is '0 2 * * *'."""
        docs = _helm_template("--set", "postgresql.backup.enabled=true")
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        self.assertEqual(len(cronjobs), 1)
        self.assertEqual(cronjobs[0]["spec"]["schedule"], "0 2 * * *")

    def test_backup_concurrency_policy_forbid(self):
        """CronJob concurrencyPolicy must be Forbid to prevent overlapping backups."""
        docs = _helm_template("--set", "postgresql.backup.enabled=true")
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        self.assertEqual(len(cronjobs), 1)
        self.assertEqual(cronjobs[0]["spec"]["concurrencyPolicy"], "Forbid")

    def test_backup_uses_postgres_image(self):
        """Backup container image comes from postgresql.image values."""
        docs = _helm_template(
            "--set", "postgresql.backup.enabled=true",
            "--set", "postgresql.image.repository=myregistry/postgres",
            "--set", "postgresql.image.tag=15-bullseye",
        )
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        containers = (
            cronjobs[0]["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"]
        )
        self.assertEqual(containers[0]["image"], "myregistry/postgres:15-bullseye")

    def test_backup_timeout_configurable(self):
        """activeDeadlineSeconds reflects postgresql.backup.timeoutSeconds."""
        docs = _helm_template(
            "--set", "postgresql.backup.enabled=true",
            "--set", "postgresql.backup.timeoutSeconds=7200",
        )
        cronjobs = _find_resources(docs, "CronJob", "postgres-backup")
        job_spec = cronjobs[0]["spec"]["jobTemplate"]["spec"]
        self.assertEqual(job_spec["activeDeadlineSeconds"], 7200)


@unittest.skipUnless(HELM_AVAILABLE, "helm CLI not found on PATH")
class TestHelmRedisSentinel(unittest.TestCase):
    """Tests for the Redis Sentinel Helm template."""

    def test_sentinel_disabled_by_default(self):
        """With default values, no sentinel resources should be rendered."""
        docs = _helm_template()
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        sentinel_svc = _find_resources(docs, "Service", "redis-sentinel")
        replica_sts = _find_resources(docs, "StatefulSet", "redis-replica")
        self.assertEqual(
            len(sentinel_sts), 0,
            "No redis-sentinel StatefulSet should be rendered by default"
        )
        self.assertEqual(
            len(sentinel_svc), 0,
            "No redis-sentinel Service should be rendered by default"
        )
        self.assertEqual(
            len(replica_sts), 0,
            "No redis-replica StatefulSet should be rendered by default"
        )

    def test_sentinel_enabled(self):
        """When redis.sentinel.enabled=true, sentinel StatefulSet and Service are present."""
        docs = _helm_template("--set", "redis.sentinel.enabled=true")
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        sentinel_svc = _find_resources(docs, "Service", "redis-sentinel")
        self.assertGreaterEqual(
            len(sentinel_sts), 1,
            "At least one redis-sentinel StatefulSet should be rendered"
        )
        self.assertGreaterEqual(
            len(sentinel_svc), 1,
            "At least one redis-sentinel Service should be rendered"
        )

    def test_sentinel_replica_statefulset(self):
        """When sentinel is enabled, a redis-replica StatefulSet with replicaof is rendered."""
        docs = _helm_template("--set", "redis.sentinel.enabled=true")
        replica_sts = _find_resources(docs, "StatefulSet", "redis-replica")
        self.assertEqual(
            len(replica_sts), 1,
            "Exactly one redis-replica StatefulSet should be rendered"
        )
        # Verify the replica command includes replicaof
        containers = replica_sts[0]["spec"]["template"]["spec"]["containers"]
        self.assertEqual(len(containers), 1)
        # Command is ["sh", "-c", "<script>"] -- check the script
        command_parts = containers[0]["command"]
        full_command = " ".join(str(p) for p in command_parts)
        self.assertIn(
            "replicaof", full_command,
            "redis-replica container command must include 'replicaof'"
        )

    def test_sentinel_master_name(self):
        """Sentinel config uses the master name from values."""
        custom_master = "my-ocr-master"
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
            "--set", f"redis.sentinel.masterName={custom_master}",
        )
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        self.assertEqual(len(sentinel_sts), 1)
        # The sentinel container command should contain the master name
        containers = sentinel_sts[0]["spec"]["template"]["spec"]["containers"]
        sentinel_container = None
        for c in containers:
            if c["name"] == "sentinel":
                sentinel_container = c
                break
        self.assertIsNotNone(sentinel_container, "Sentinel container not found")
        command_parts = sentinel_container["command"]
        full_command = " ".join(str(p) for p in command_parts)
        self.assertIn(
            f"sentinel monitor {custom_master}",
            full_command,
            f"Sentinel command must contain 'sentinel monitor {custom_master}'"
        )

    def test_sentinel_replicas_default(self):
        """Default sentinelReplicas=3 controls Sentinel StatefulSet replicas."""
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
        )
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        self.assertEqual(len(sentinel_sts), 1)
        self.assertEqual(
            sentinel_sts[0]["spec"]["replicas"], 3,
            "Sentinel StatefulSet should default to 3 replicas (sentinelReplicas)"
        )

    def test_sentinel_replicas_override(self):
        """Setting sentinelReplicas=5 overrides Sentinel StatefulSet replicas."""
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
            "--set", "redis.sentinel.sentinelReplicas=5",
        )
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        self.assertEqual(len(sentinel_sts), 1)
        self.assertEqual(sentinel_sts[0]["spec"]["replicas"], 5)

    def test_sentinel_headless_service(self):
        """Sentinel Service must be headless (clusterIP: None)."""
        docs = _helm_template("--set", "redis.sentinel.enabled=true")
        sentinel_svc = _find_resources(docs, "Service", "redis-sentinel")
        self.assertGreaterEqual(len(sentinel_svc), 1)
        self.assertEqual(
            sentinel_svc[0]["spec"]["clusterIP"], "None",
            "Sentinel service must be headless (clusterIP: None)"
        )

    def test_sentinel_service_port(self):
        """Sentinel Service exposes port 26379."""
        docs = _helm_template("--set", "redis.sentinel.enabled=true")
        sentinel_svc = _find_resources(docs, "Service", "redis-sentinel")
        self.assertGreaterEqual(len(sentinel_svc), 1)
        ports = sentinel_svc[0]["spec"]["ports"]
        port_numbers = [p["port"] for p in ports]
        self.assertIn(26379, port_numbers, "Sentinel service must expose port 26379")

    def test_sentinel_down_after_ms_configurable(self):
        """Custom down-after-milliseconds appears in the sentinel command."""
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
            "--set", "redis.sentinel.downAfterMs=10000",
        )
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        containers = sentinel_sts[0]["spec"]["template"]["spec"]["containers"]
        sentinel_container = [c for c in containers if c["name"] == "sentinel"][0]
        full_command = " ".join(str(p) for p in sentinel_container["command"])
        self.assertIn(
            "down-after-milliseconds", full_command,
        )
        self.assertIn("10000", full_command)

    def test_sentinel_failover_timeout_configurable(self):
        """Custom failover-timeout appears in the sentinel command."""
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
            "--set", "redis.sentinel.failoverTimeoutMs=30000",
        )
        sentinel_sts = _find_resources(docs, "StatefulSet", "redis-sentinel")
        containers = sentinel_sts[0]["spec"]["template"]["spec"]["containers"]
        sentinel_container = [c for c in containers if c["name"] == "sentinel"][0]
        full_command = " ".join(str(p) for p in sentinel_container["command"])
        self.assertIn("failover-timeout", full_command)
        self.assertIn("30000", full_command)

    def test_replica_count_configurable(self):
        """redis.sentinel.replicas controls the redis-replica StatefulSet replica count."""
        docs = _helm_template(
            "--set", "redis.sentinel.enabled=true",
            "--set", "redis.sentinel.replicas=3",
        )
        replica_sts = _find_resources(docs, "StatefulSet", "redis-replica")
        self.assertEqual(len(replica_sts), 1)
        self.assertEqual(
            replica_sts[0]["spec"]["replicas"], 3,
            "redis-replica StatefulSet should have 3 replicas"
        )


class TestRedisSentinelSettings(TestCase):
    """Tests for Redis Sentinel configuration in coordinator/coordinator/settings.py.

    These tests reload the production settings module with patched env vars
    to verify the conditional Sentinel transport options logic.
    """

    def _env_base(self):
        """Return base environment variables required by settings.py."""
        return {
            'DJANGO_DEBUG': 'True',
            'DATABASE_URL': 'postgres://testuser:testpass@localhost:5432/testdb',
            'CELERY_BROKER_URL': 'amqp://guest:guest@localhost:5672//',
            'DEPLOYMENT_ENV': 'development',
        }

    def _reload_settings(self):
        """Reload the production settings module to pick up env var changes.

        Before reloading, we must delete the conditionally-defined attribute
        CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS because importlib.reload does
        NOT clear stale module-level names that are absent from the new execution.
        """
        import coordinator.settings as settings_mod
        if hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS'):
            del settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
        importlib.reload(settings_mod)
        return settings_mod

    def test_sentinel_config_not_set_by_default(self):
        """Without REDIS_SENTINEL_MASTER_NAME, transport options are not defined."""
        env = self._env_base()
        # Ensure sentinel env vars are absent
        env.pop('REDIS_SENTINEL_MASTER_NAME', None)
        env.pop('REDIS_SENTINEL_PASSWORD', None)
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            self.assertFalse(
                hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS'),
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS should not be defined "
                "when REDIS_SENTINEL_MASTER_NAME is not set"
            )

    def test_sentinel_config_set_with_master_name(self):
        """When REDIS_SENTINEL_MASTER_NAME=ocr-master, transport options include master_name."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            self.assertTrue(
                hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS'),
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS must be defined "
                "when REDIS_SENTINEL_MASTER_NAME is set"
            )
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            self.assertEqual(transport_opts['master_name'], 'ocr-master')

    def test_sentinel_config_includes_password(self):
        """When both REDIS_SENTINEL_MASTER_NAME and REDIS_SENTINEL_PASSWORD are set,
        sentinel_kwargs.password is configured."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        env['REDIS_SENTINEL_PASSWORD'] = 's3cret-sentinel'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            self.assertIn(
                'sentinel_kwargs', transport_opts,
                "sentinel_kwargs must be present when REDIS_SENTINEL_PASSWORD is set"
            )
            self.assertEqual(
                transport_opts['sentinel_kwargs']['password'],
                's3cret-sentinel',
            )

    def test_sentinel_no_password_omits_sentinel_kwargs(self):
        """With only master name (no password), sentinel_kwargs is NOT in transport options."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        # Do not set REDIS_SENTINEL_PASSWORD
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            self.assertNotIn(
                'sentinel_kwargs', transport_opts,
                "sentinel_kwargs must NOT be present when "
                "REDIS_SENTINEL_PASSWORD is not set"
            )

    def test_sentinel_custom_master_name(self):
        """Transport options reflect the exact master name from the env var."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'custom-redis-primary'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            self.assertEqual(transport_opts['master_name'], 'custom-redis-primary')

    def test_sentinel_result_backend_url(self):
        """When Sentinel is used, CELERY_RESULT_BACKEND can be a sentinel:// URL."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        env['CELERY_RESULT_BACKEND'] = 'sentinel://sentinel1:26379/0'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            self.assertEqual(
                settings_mod.CELERY_RESULT_BACKEND,
                'sentinel://sentinel1:26379/0',
            )
            # Transport options should still be set
            self.assertTrue(
                hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS')
            )
