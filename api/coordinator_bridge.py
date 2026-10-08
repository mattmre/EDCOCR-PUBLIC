"""Initialize the intentionally co-packaged Django coordinator boundary."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path

_setup_lock = threading.Lock()
_setup_done = False


def ensure_coordinator_django() -> None:
    """Set up Django once before FastAPI touches coordinator-owned models."""
    global _setup_done
    if _setup_done:
        return
    with _setup_lock:
        if _setup_done:
            return
        source_coordinator = Path(__file__).resolve().parent.parent / "coordinator"
        source_path = str(source_coordinator)
        inserted_source_path = False
        if source_coordinator.is_dir() and source_path not in sys.path:
            sys.path.insert(0, source_path)
            inserted_source_path = True
        settings_was_set = "DJANGO_SETTINGS_MODULE" in os.environ
        settings_module = os.environ.get(
            "DJANGO_SETTINGS_MODULE", "coordinator.settings"
        )
        os.environ["DJANGO_SETTINGS_MODULE"] = settings_module
        try:
            # Import settings before django.setup() enters Apps.populate().
            # Production settings intentionally fail closed when required
            # secrets are absent.  Performing that check first prevents an
            # optional standalone-API enrichment attempt from leaving Django's
            # global app registry half-populated.
            importlib.import_module(settings_module)
        except ImportError as exc:
            if not settings_was_set:
                os.environ.pop("DJANGO_SETTINGS_MODULE", None)
            if inserted_source_path:
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
            raise RuntimeError(
                "The API image lacks its required coordinator runtime"
            ) from exc
        except Exception:
            if not settings_was_set:
                os.environ.pop("DJANGO_SETTINGS_MODULE", None)
            if inserted_source_path:
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
            raise
        try:
            import django

            django.setup()
        except ImportError as exc:
            if not settings_was_set:
                os.environ.pop("DJANGO_SETTINGS_MODULE", None)
            if inserted_source_path:
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
            raise RuntimeError(
                "The API image lacks its required coordinator runtime"
            ) from exc
        except Exception:
            if not settings_was_set:
                os.environ.pop("DJANGO_SETTINGS_MODULE", None)
            if inserted_source_path:
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
            raise
        # Bind shared_task proxies to the configured coordinator app. Merely
        # initializing Django leaves them on Celery's unsigned default app,
        # which production workers correctly reject.
        from coordinator.celery import app as _coordinator_celery_app  # noqa: F401

        _setup_done = True
