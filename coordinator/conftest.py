"""Pytest configuration for Django coordinator tests.

The test settings module (coordinator.settings_test) is configured in pytest.ini.
It uses SQLite in-memory and synchronous Celery, so no external services are needed.
"""

import shutil
import tempfile
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """Remove only the default process-scoped coordinator test artifact root."""
    from django.conf import settings

    root = Path(settings.NFS_ROOT).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if root.parent == temp_root and root.name.startswith(
        "ocr-local-coordinator-tests-"
    ):
        shutil.rmtree(root, ignore_errors=True)
