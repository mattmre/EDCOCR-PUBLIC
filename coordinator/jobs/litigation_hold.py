"""Centralized LITIGATION_HOLD enforcement for all data-deletion paths.

When the ``LITIGATION_HOLD`` environment variable is set to a truthy value
(``1``, ``true``, or ``yes``, case-insensitive), **all** automated and
manual data-deletion operations must be blocked.  This module provides a
single source of truth so every management command and Celery task uses
the same check logic.

Environment variable:
    LITIGATION_HOLD
        Set to ``1``, ``true``, or ``yes`` to freeze all data deletions.
        Default (unset or any other value): deletions are allowed.

Usage in management commands::

    from jobs.litigation_hold import check_litigation_hold

    class Command(BaseCommand):
        def handle(self, *args, **options):
            if check_litigation_hold(self):
                return
            # ... proceed with deletions

Usage in Celery tasks::

    from jobs.litigation_hold import is_litigation_hold_active

    @shared_task
    def my_cleanup_task():
        if is_litigation_hold_active():
            return {"status": "skipped", "litigation_hold": True}
        # ... proceed with deletions
"""

import os


def is_litigation_hold_active():
    """Return True when LITIGATION_HOLD env var is set to a truthy value.

    Recognised truthy values (case-insensitive): ``1``, ``true``, ``yes``.
    """
    return os.environ.get("LITIGATION_HOLD", "").lower() in ("1", "true", "yes")


def check_litigation_hold(command):
    """Check litigation hold and emit an error via the command's stderr.

    Parameters
    ----------
    command : django.core.management.base.BaseCommand
        The management command instance (used for ``self.stderr.write``).

    Returns
    -------
    bool
        ``True`` if the hold is active and the caller should abort.
        ``False`` if the caller may proceed.
    """
    if is_litigation_hold_active():
        command.stderr.write(
            "LITIGATION_HOLD is active -- all data deletions are blocked."
        )
        return True
    return False
