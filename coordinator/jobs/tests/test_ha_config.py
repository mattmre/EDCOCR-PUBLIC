"""Tests for Phase 7D Infrastructure HA configuration.

Tests broker resilience settings in celery.py, quorum queue toggles,
Redis Sentinel support in settings.py, and task routing correctness.

Run with: cd coordinator && python -m pytest jobs/tests/test_ha_config.py -v
"""

import importlib
import os
from unittest.mock import patch

from django.test import TestCase


class TestCeleryBrokerResilience(TestCase):
    """Tests that verify celery.py has the correct resilience settings."""

    def setUp(self):
        # Import the Celery app; the module is already loaded so we
        # just grab the singleton app instance.
        from coordinator.celery import app
        self.app = app

    def test_broker_connection_retry_on_startup(self):
        """Celery app must retry broker connections on startup."""
        assert self.app.conf.broker_connection_retry_on_startup is True

    def test_broker_connection_max_retries(self):
        """Celery app must limit broker connection retries to 10."""
        assert self.app.conf.broker_connection_max_retries == 10

    def test_broker_transport_confirm_publish(self):
        """Broker transport options must include confirm_publish: True."""
        transport_opts = self.app.conf.broker_transport_options
        assert isinstance(transport_opts, dict)
        assert transport_opts.get('confirm_publish') is True

    def test_task_acks_late(self):
        """Tasks must be acknowledged only after execution (acks_late)."""
        assert self.app.conf.task_acks_late is True

    def test_task_reject_on_worker_lost(self):
        """Tasks must be rejected (requeued) when a worker is lost."""
        assert self.app.conf.task_reject_on_worker_lost is True

    def test_worker_prefetch_multiplier(self):
        """Worker prefetch multiplier must be 1 for fair scheduling."""
        assert self.app.conf.worker_prefetch_multiplier == 1


class TestQuorumQueues(TestCase):
    """Tests for the quorum queue toggle controlled by CELERY_USE_QUORUM_QUEUES."""

    def test_quorum_queues_disabled_by_default(self):
        """Without env var, standard queues are used (no x-queue-type argument)."""
        env_clean = {
            k: v for k, v in os.environ.items()
            if k != 'CELERY_USE_QUORUM_QUEUES'
        }
        with patch.dict(os.environ, env_clean, clear=True):
            import coordinator.celery as celery_mod
            importlib.reload(celery_mod)
            try:
                # When quorum is disabled, task_queues is either not set
                # or the queues do not have x-queue-type in their arguments.
                task_queues = celery_mod.app.conf.task_queues
                if task_queues:
                    for q in task_queues:
                        queue_args = getattr(q, 'queue_arguments', {}) or {}
                        assert 'x-queue-type' not in queue_args, (
                            f"Queue '{q.name}' should NOT have x-queue-type "
                            f"when CELERY_USE_QUORUM_QUEUES is not set"
                        )
            finally:
                # Restore the module to its original state
                importlib.reload(celery_mod)

    def test_quorum_queues_enabled(self):
        """Setting CELERY_USE_QUORUM_QUEUES=true adds x-queue-type: quorum to all queues."""
        with patch.dict(os.environ, {'CELERY_USE_QUORUM_QUEUES': 'true'}):
            import coordinator.celery as celery_mod
            importlib.reload(celery_mod)
            try:
                task_queues = celery_mod.app.conf.task_queues
                assert task_queues is not None, (
                    "task_queues must be set when quorum queues are enabled"
                )
                assert len(task_queues) == 6, (
                    f"Expected 6 quorum queues, got {len(task_queues)}"
                )
                for q in task_queues:
                    queue_args = getattr(q, 'queue_arguments', {}) or {}
                    assert queue_args.get('x-queue-type') == 'quorum', (
                        f"Queue '{q.name}' must have x-queue-type: quorum"
                    )
            finally:
                importlib.reload(celery_mod)

    def test_quorum_queue_names(self):
        """When enabled, quorum queues must include all declared coordinator and worker queues."""
        expected_names = {
            'coordinator',
            'ocr_gpu',
            'ocr_cpu',
            'cpu_general',
            'nlp_general',
            'ocr_layoutlm',
        }
        with patch.dict(os.environ, {'CELERY_USE_QUORUM_QUEUES': 'true'}):
            import coordinator.celery as celery_mod
            importlib.reload(celery_mod)
            try:
                task_queues = celery_mod.app.conf.task_queues
                actual_names = {q.name for q in task_queues}
                assert actual_names == expected_names, (
                    f"Expected queue names {expected_names}, got {actual_names}"
                )
            finally:
                importlib.reload(celery_mod)


class TestRedisSentinelConfig(TestCase):
    """Tests for the Sentinel-aware result backend configuration in settings.py."""

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

    def _env_base(self):
        """Return base environment variables required by settings.py."""
        return {
            'DJANGO_DEBUG': 'True',
            'DATABASE_URL': 'postgres://testuser:testpass@localhost:5432/testdb',
            'CELERY_BROKER_URL': 'amqp://guest:guest@localhost:5672//',
            'DEPLOYMENT_ENV': 'development',
        }

    def test_sentinel_not_configured_by_default(self):
        """Without REDIS_SENTINEL_MASTER_NAME, transport options are not set."""
        env = self._env_base()
        # Explicitly remove Sentinel env vars
        env.pop('REDIS_SENTINEL_MASTER_NAME', None)
        env.pop('REDIS_SENTINEL_PASSWORD', None)
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert not hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS'), (
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS should not be defined "
                "when REDIS_SENTINEL_MASTER_NAME is not set"
            )

    def test_sentinel_master_name_configured(self):
        """With REDIS_SENTINEL_MASTER_NAME set, transport options include master_name."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert hasattr(settings_mod, 'CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS'), (
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS must be defined "
                "when REDIS_SENTINEL_MASTER_NAME is set"
            )
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert transport_opts['master_name'] == 'ocr-master'

    def test_sentinel_password_configured(self):
        """With both master name and password, sentinel_kwargs.password is set."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        env['REDIS_SENTINEL_PASSWORD'] = 's3cret-sentinel'
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert 'sentinel_kwargs' in transport_opts, (
                "sentinel_kwargs must be present when REDIS_SENTINEL_PASSWORD is set"
            )
            assert transport_opts['sentinel_kwargs']['password'] == 's3cret-sentinel'

    def test_sentinel_no_password(self):
        """With only master name (no password), sentinel_kwargs is NOT in transport options."""
        env = self._env_base()
        env['REDIS_SENTINEL_MASTER_NAME'] = 'ocr-master'
        # Do not set REDIS_SENTINEL_PASSWORD
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert 'sentinel_kwargs' not in transport_opts, (
                "sentinel_kwargs must NOT be present when "
                "REDIS_SENTINEL_PASSWORD is not set"
            )


class TestTaskRouting(TestCase):
    """Tests for queue routing correctness in celery.py _route_task function.

    task_routes is now a tuple containing the _route_task callable (rather
    than a static dict) to support per-GPU queue affinity round-robin.
    """

    def setUp(self):
        from coordinator.celery import _route_task
        self.route = _route_task

    def _resolve(self, task_name):
        """Call the router and return the resolved queue name."""
        result = self.route(task_name, args=(), kwargs={}, options={})
        assert result is not None, (
            f"Task '{task_name}' must be recognised by _route_task"
        )
        return result['queue']

    def test_coordinator_tasks_route_to_coordinator_queue(self):
        """ingest_document, assemble_document, finalize_job route to coordinator queue."""
        coordinator_tasks = [
            'jobs.tasks.ingest_document',
            'jobs.tasks.assemble_document',
            'jobs.tasks.finalize_job',
        ]
        for task_name in coordinator_tasks:
            queue = self._resolve(task_name)
            assert queue == 'coordinator', (
                f"Task '{task_name}' must route to 'coordinator' queue, "
                f"got '{queue}'"
            )

    def test_gpu_tasks_route_to_gpu_queue(self):
        """process_document, process_page route to ocr_gpu queue (default, no per-GPU)."""
        gpu_tasks = [
            'jobs.tasks.process_document',
            'jobs.tasks.process_page',
        ]
        for task_name in gpu_tasks:
            queue = self._resolve(task_name)
            assert queue.startswith('ocr_gpu'), (
                f"Task '{task_name}' must route to an ocr_gpu* queue, "
                f"got '{queue}'"
            )

    def test_cpu_tasks_route_to_cpu_queue(self):
        """compress_pdf, extract_entities route to cpu_general queue."""
        cpu_tasks = [
            'jobs.tasks.compress_pdf',
            'jobs.tasks.extract_entities',
        ]
        for task_name in cpu_tasks:
            queue = self._resolve(task_name)
            assert queue == 'cpu_general', (
                f"Task '{task_name}' must route to 'cpu_general' queue, "
                f"got '{queue}'"
            )

    def test_unknown_task_returns_none(self):
        """An unrecognised task name should return None (Celery default routing)."""
        result = self.route('some.unknown.task', args=(), kwargs={}, options={})
        assert result is None, (
            f"Unknown task should return None, got {result}"
        )
