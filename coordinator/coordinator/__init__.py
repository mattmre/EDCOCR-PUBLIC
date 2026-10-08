"""Coordinator Django package bootstrap.

The Celery app is exposed lazily so importing a pure coordinator namespace
such as ``coordinator.federation.custody`` does not initialize Django/Celery
or require production credentials.  Worker entry points that import
``coordinator.celery`` or ``from coordinator import celery_app`` still get the
same application object.
"""

from pathlib import Path

_PROJECT_DIR = str(Path(__file__).resolve().parents[1])
if _PROJECT_DIR not in __path__:  # type: ignore[name-defined]
    __path__.append(_PROJECT_DIR)  # type: ignore[name-defined]

__all__ = ("celery_app",)


def __getattr__(name: str):
    """Resolve the conventional Celery export only when requested."""
    if name == "celery_app":
        from .celery import app

        return app
    raise AttributeError(name)
