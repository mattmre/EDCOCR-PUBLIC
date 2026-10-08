"""Byte admission for queued, decompressed page rasters."""

from __future__ import annotations

import threading
from collections.abc import Callable


class RasterByteBudget:
    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("raster byte budget must be positive")
        self.limit = int(limit)
        self.used = 0
        self._condition = threading.Condition()

    def acquire(
        self,
        amount: int,
        *,
        wait: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        if amount <= 0 or amount > self.limit:
            raise ValueError("raster exceeds queue byte budget")
        with self._condition:
            while self.used + amount > self.limit:
                if not wait or (cancelled is not None and cancelled()):
                    return False
                self._condition.wait(timeout=0.2)
            self.used += amount
            return True

    def release(self, amount: int) -> None:
        with self._condition:
            if amount <= 0 or amount > self.used:
                raise RuntimeError("raster byte budget underflow")
            self.used -= amount
            self._condition.notify_all()
