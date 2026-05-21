import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class JobsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'jobs'
    verbose_name = 'OCR Jobs'

    def ready(self):
        import jobs.signals  # noqa: F401 — register Celery signal handlers

        # Optional startup credential validation (production only)
        if os.environ.get('DEPLOYMENT_ENV', '').strip().lower() == 'production':
            self._validate_credentials()

    @staticmethod
    def _validate_credentials():
        """Log warnings for missing or placeholder credentials at startup.

        Never blocks startup -- credential issues are logged as warnings
        so operators can detect misconfiguration from the log stream.
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
