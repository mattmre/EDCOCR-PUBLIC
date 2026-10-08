import importlib
import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class JobsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "jobs"
    verbose_name = "OCR Jobs"

    def ready(self):
        # Django publishers (admin actions, management commands, API bridge)
        # must bind shared_task proxies to the same signed coordinator app as
        # Beat/workers.  Importing only Django otherwise leaves Celery's
        # unsigned default app current, so a broker send can be recorded as
        # published and then rejected by a correctly configured worker.
        importlib.import_module("jobs.signals")
        coordinator_celery_app = importlib.import_module("coordinator.celery").app

        # ``autodiscover_tasks()`` is lazy by default.  Workers finalize it as
        # part of Celery boot, but a plain Django process can publish from an
        # admin request before that signal fires.  Force discovery now that
        # Django models are loaded so outbox lookups cannot fail after the DB
        # contract has committed.
        coordinator_celery_app.autodiscover_tasks(force=True)
        coordinator_celery_app.autodiscover_tasks(
            ["jobs"], related_name="tasks_layoutlm", force=True
        )
        coordinator_celery_app.loader.import_task_module("jobs.tasks_translation_batch")

        # Optional startup credential validation (production only)
        if os.environ.get("DEPLOYMENT_ENV", "").strip().lower() == "production":
            # PII fields and presigned capability payloads share the encrypted
            # field primitive. Fail startup before accepting work if its key is
            # missing/invalid instead of failing on the first database write.
            from jobs.pii_encryption import get_encryptor

            get_encryptor()
            self._validate_credentials()

    @staticmethod
    def _validate_credentials():
        """Log warnings for missing or placeholder credentials at startup.

        Non-encryption credential issues remain warnings; the encryption key
        is validated separately and fail-closed before this method is called.
        """
        try:
            from coordinator.credential_bridge import validate_credentials

            report = validate_credentials(strict=True)
            if not report.passed:
                for result in report.errors:
                    logger.warning(
                        "Credential issue at startup: [%s] %s",
                        result.status,
                        result.message,
                    )
            else:
                logger.info(
                    "Startup credential validation passed: %s",
                    report.summary(),
                )
        except Exception:
            # Never block startup on credential validation failures
            logger.debug(
                "Credential validation skipped (not available)",
                exc_info=True,
            )
