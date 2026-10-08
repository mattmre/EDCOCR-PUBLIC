"""Tests for Phase 7D Infrastructure HA configuration.

Tests broker resilience settings in celery.py, quorum queue toggles,
Redis Sentinel support in settings.py, and task routing correctness.

Run with: cd coordinator && python -m pytest jobs/tests/test_ha_config.py -v
"""

import importlib
import os
import ssl
from types import SimpleNamespace
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
        assert transport_opts.get("confirm_publish") is True

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
            k: v for k, v in os.environ.items() if k != "CELERY_USE_QUORUM_QUEUES"
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
                        queue_args = getattr(q, "queue_arguments", {}) or {}
                        assert "x-queue-type" not in queue_args, (
                            f"Queue '{q.name}' should NOT have x-queue-type "
                            f"when CELERY_USE_QUORUM_QUEUES is not set"
                        )
            finally:
                # Restore the module to its original state
                importlib.reload(celery_mod)

    def test_quorum_queues_enabled(self):
        """Setting CELERY_USE_QUORUM_QUEUES=true adds x-queue-type: quorum to all queues."""
        with patch.dict(os.environ, {"CELERY_USE_QUORUM_QUEUES": "true"}):
            import coordinator.celery as celery_mod

            importlib.reload(celery_mod)
            try:
                task_queues = celery_mod.app.conf.task_queues
                assert (
                    task_queues is not None
                ), "task_queues must be set when quorum queues are enabled"
                assert (
                    len(task_queues) == 10
                ), f"Expected 10 quorum queues, got {len(task_queues)}"
                for q in task_queues:
                    queue_args = getattr(q, "queue_arguments", {}) or {}
                    assert (
                        queue_args.get("x-queue-type") == "quorum"
                    ), f"Queue '{q.name}' must have x-queue-type: quorum"
                    # Quorum + classic x-max-priority is invalid on RabbitMQ
                    assert "x-max-priority" not in queue_args, (
                        f"Queue '{q.name}' must NOT set x-max-priority with quorum "
                        f"(queue declaration compatibility)"
                    )
            finally:
                importlib.reload(celery_mod)

    def test_quorum_queue_names(self):
        """When enabled, quorum queues must include all declared coordinator and worker queues."""
        expected_names = {
            "coordinator",
            "ocr_gpu",
            "ocr_cpu",
            "ocr_tesseract",
            "ocr_render",
            "ocr_finalize",
            "cpu_general",
            "nlp_general",
            "ocr_layoutlm",
            "translation_batch",
        }
        with patch.dict(os.environ, {"CELERY_USE_QUORUM_QUEUES": "true"}):
            import coordinator.celery as celery_mod

            importlib.reload(celery_mod)
            try:
                task_queues = celery_mod.app.conf.task_queues
                actual_names = {q.name for q in task_queues}
                assert (
                    actual_names == expected_names
                ), f"Expected queue names {expected_names}, got {actual_names}"
            finally:
                importlib.reload(celery_mod)

    def test_strict_per_gpu_routes_are_declared_as_quorum(self):
        """The quorum + strict-affinity cross-product must not create classic queues."""
        env = {
            "CELERY_USE_QUORUM_QUEUES": "true",
            "ENABLE_PER_GPU_QUEUES": "true",
            "OCR_PER_GPU_STRICT": "true",
            "GPU_COUNT": "2",
        }
        import coordinator.celery as celery_mod

        try:
            with patch.dict(os.environ, env):
                importlib.reload(celery_mod)
                task_queues = celery_mod.app.conf.task_queues
                declared = {queue.name: queue for queue in task_queues}
                assert {"ocr_gpu_0", "ocr_gpu_1"} <= declared.keys()
                assert celery_mod.app.conf.task_default_queue == "coordinator"
                assert celery_mod.app.conf.task_create_missing_queues is False
                for name in ("ocr_gpu_0", "ocr_gpu_1"):
                    assert declared[name].queue_arguments == {"x-queue-type": "quorum"}

                routed = {
                    celery_mod._route_task(
                        "jobs.tasks.process_page", (), {}, {}, task=None
                    )["queue"]
                    for _ in range(2)
                }
                assert routed == {"ocr_gpu_0", "ocr_gpu_1"}
                assert routed <= declared.keys()
        finally:
            importlib.reload(celery_mod)

    def test_federated_strict_gpu_queue_is_declared_as_quorum(self):
        """A peer suffix must not turn fail-closed quorum routing into QueueNotFound."""
        env = {
            "CELERY_USE_QUORUM_QUEUES": "true",
            "ENABLE_PER_GPU_QUEUES": "true",
            "OCR_PER_GPU_STRICT": "true",
            "GPU_COUNT": "2",
        }
        import coordinator.celery as celery_mod

        class PeerRouter:
            @staticmethod
            def select_cluster(*, task_name, base_queue, kwargs):
                del task_name, kwargs
                return SimpleNamespace(queue_name=f"{base_queue}.peer-a")

        try:
            with (
                patch.dict(os.environ, env),
                patch.object(
                    celery_mod, "_get_federation_router", return_value=PeerRouter()
                ),
            ):
                importlib.reload(celery_mod)
                with patch.object(
                    celery_mod, "_get_federation_router", return_value=PeerRouter()
                ):
                    result = celery_mod._route_task(
                        "jobs.tasks.process_page", (), {}, {}, task=None
                    )
                    destination = result["queue"]
                    assert destination.startswith("ocr_gpu_")
                    assert destination.endswith(".peer-a")
                    declared = celery_mod.app.amqp.queues[destination]
                    assert declared.queue_arguments == {"x-queue-type": "quorum"}
                    routed = celery_mod.app.amqp.Router().route(
                        {}, "jobs.tasks.process_page", args=(), kwargs={}
                    )
                    assert routed["queue"].name.endswith(".peer-a")
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

        for attr in (
            "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS",
            "CELERY_BROKER_USE_SSL",
            "CELERY_REDIS_BACKEND_USE_SSL",
        ):
            if hasattr(settings_mod, attr):
                delattr(settings_mod, attr)
        importlib.reload(settings_mod)
        return settings_mod

    def _env_base(self):
        """Return base environment variables required by settings.py."""
        # Local fixture URLs only — not production credentials.
        return {
            "DJANGO_DEBUG": "True",
            "DATABASE_URL": "postgres://testuser:testpass@localhost:5432/testdb",  # pragma: allowlist secret
            "CELERY_BROKER_URL": "amqp://guest:guest@localhost:5672//",  # pragma: allowlist secret
            "DEPLOYMENT_ENV": "development",
        }

    def test_sentinel_not_configured_by_default(self):
        """Without REDIS_SENTINEL_MASTER_NAME, transport options are not set."""
        env = self._env_base()
        # Explicitly remove Sentinel env vars
        env.pop("REDIS_SENTINEL_MASTER_NAME", None)
        env.pop("REDIS_SENTINEL_PASSWORD", None)
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert not hasattr(
                settings_mod, "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS"
            ), (
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS should not be defined "
                "when REDIS_SENTINEL_MASTER_NAME is not set"
            )

    def test_sentinel_master_name_configured(self):
        """With REDIS_SENTINEL_MASTER_NAME set, transport options include master_name."""
        env = self._env_base()
        env["REDIS_SENTINEL_MASTER_NAME"] = "ocr-master"
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert hasattr(settings_mod, "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS"), (
                "CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS must be defined "
                "when REDIS_SENTINEL_MASTER_NAME is set"
            )
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert transport_opts["master_name"] == "ocr-master"

    def test_sentinel_password_configured(self):
        """With both master name and password, sentinel_kwargs.password is set."""
        env = self._env_base()
        env["REDIS_SENTINEL_MASTER_NAME"] = "ocr-master"
        fixture_pw = "test-sentinel-fixture-not-prod"  # pragma: allowlist secret
        env["REDIS_SENTINEL_PASSWORD"] = fixture_pw
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert (
                "sentinel_kwargs" in transport_opts
            ), "sentinel_kwargs must be present when REDIS_SENTINEL_PASSWORD is set"
            assert transport_opts["sentinel_kwargs"]["password"] == fixture_pw

    def test_sentinel_no_password(self):
        """With only master name (no password), sentinel_kwargs is NOT in transport options."""
        env = self._env_base()
        env["REDIS_SENTINEL_MASTER_NAME"] = "ocr-master"
        # Do not set REDIS_SENTINEL_PASSWORD
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            transport_opts = settings_mod.CELERY_RESULT_BACKEND_TRANSPORT_OPTIONS
            assert "sentinel_kwargs" not in transport_opts, (
                "sentinel_kwargs must NOT be present when "
                "REDIS_SENTINEL_PASSWORD is not set"
            )

    def test_tls_flags_upgrade_urls_and_require_ca_verification(self):
        """TLS toggles must reach both URL schemes and client settings."""
        env = self._env_base()
        env.update(
            {
                # Public RabbitMQ defaults used only as a URL-shape fixture.
                "CELERY_BROKER_URL": "amqp://guest:guest@localhost:5672//",  # pragma: allowlist secret
                "CELERY_BROKER_USE_SSL": "true",
                "REDIS_SSL": "true",
                "REDIS_URL": "redis://:password@localhost:6379/0",  # pragma: allowlist secret
                "CELERY_RESULT_BACKEND": "redis://:password@localhost:6379/1",  # pragma: allowlist secret
            }
        )
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert settings_mod.CELERY_BROKER_URL.startswith("amqps://")
            assert ":5671//" in settings_mod.CELERY_BROKER_URL
            assert settings_mod.CELERY_BROKER_USE_SSL["cert_reqs"] == ssl.CERT_REQUIRED
            assert settings_mod.REDIS_URL.startswith("rediss://")
            assert ":6380/0" in settings_mod.REDIS_URL
            assert settings_mod.CELERY_RESULT_BACKEND.startswith("rediss://")
            assert ":6380/1" in settings_mod.CELERY_RESULT_BACKEND
            assert settings_mod.CELERY_REDIS_BACKEND_USE_SSL["ssl_cert_reqs"] == (
                ssl.CERT_REQUIRED
            )
            assert settings_mod.CACHES["default"]["OPTIONS"]["ssl_ca_certs"] == (
                "/etc/tls/ca.crt"
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
        assert (
            result is not None
        ), f"Task '{task_name}' must be recognised by _route_task"
        return result["queue"]

    def test_coordinator_tasks_route_to_coordinator_queue(self):
        """ingest_document, assemble_document, finalize_job route to coordinator queue."""
        coordinator_tasks = [
            "jobs.tasks.ingest_document",
            "jobs.tasks.assemble_document",
            "jobs.tasks.finalize_job",
        ]
        for task_name in coordinator_tasks:
            queue = self._resolve(task_name)
            assert (
                queue == "coordinator"
            ), f"Task '{task_name}' must route to 'coordinator' queue, got '{queue}'"

    def test_gpu_tasks_route_to_gpu_queue(self):
        """process_document, process_page route to ocr_gpu queue (default, no per-GPU)."""
        gpu_tasks = [
            "jobs.tasks.process_document",
            "jobs.tasks.process_page",
        ]
        for task_name in gpu_tasks:
            queue = self._resolve(task_name)
            assert queue.startswith(
                "ocr_gpu"
            ), f"Task '{task_name}' must route to an ocr_gpu* queue, got '{queue}'"

    def test_cpu_tasks_route_to_cpu_queue(self):
        """compress_pdf, extract_entities route to cpu_general queue."""
        cpu_tasks = [
            "jobs.tasks.compress_pdf",
            "jobs.tasks.extract_entities",
        ]
        for task_name in cpu_tasks:
            queue = self._resolve(task_name)
            assert (
                queue == "cpu_general"
            ), f"Task '{task_name}' must route to 'cpu_general' queue, got '{queue}'"

    def test_unknown_task_returns_none(self):
        """An unrecognised task name should return None (Celery default routing)."""
        result = self.route("some.unknown.task", args=(), kwargs={}, options={})
        assert result is None, f"Unknown task should return None, got {result}"
