"""Exclusive lifetime ownership for the standalone API's local state.

Admission locks serialize individual writes; they cannot make two JobManagers
or socket registries share ownership. Acquire this lock before startup recovery.
The OS releases it after a crash. Never unlink the lock file while it is in use.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class ApiProcessOwner:
    def __init__(self, database_path: str):
        database = Path(database_path).resolve()
        self.path = database.with_name(database.name + ".api-owner.lock")
        self.fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError("API ownership lock must be a private regular file")
            if os.name == "nt":
                import msvcrt

                if metadata.st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                "Standalone API state is already owned or cannot be locked. "
                "Run one API process per store; scale coordinator OCR workers instead."
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# A timed-out writer must keep ownership until process exit. Holding the object
# explicitly also prevents a future refactor from releasing it through GC.
undrained_owners: list[ApiProcessOwner] = []
