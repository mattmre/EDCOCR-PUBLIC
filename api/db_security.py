"""Database file security helpers.

Shared utilities for securing SQLite database files on disk. Addresses
 (expert panel P2): SQLite database files must not be world-readable
because they may contain PII, job metadata, and review decisions.
"""

from __future__ import annotations

import logging
import os
import stat

logger = logging.getLogger(__name__)


def set_db_permissions(path: str) -> None:
    """Restrict a SQLite database file to owner read/write only (0o600).

    This is a best-effort helper. On POSIX systems it sets mode 0o600 so
    that only the owning user can read or write the database. On Windows
    ``os.chmod`` has limited effect (it can only toggle the read-only
    flag), which is acceptable because Windows ACLs are expected to
    handle access control instead.

    Any ``OSError`` (missing file, permission denied, unsupported
    platform) is logged at debug level and swallowed -- permission
    tightening must never break database initialization.

    Args:
        path: Absolute or relative path to the SQLite database file.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        logger.debug(
            "Could not set 0o600 permissions on %s: %s", path, exc
        )
