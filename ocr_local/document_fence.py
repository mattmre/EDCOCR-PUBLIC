"""Per-document producer obligations for safe early OCR finalization."""

from __future__ import annotations

import threading


class DocumentCompletionFence:
    """Track one immutable source generation through nested queue stages.

    A producer reserves its child obligation before releasing its own. Once
    scheduler admission closes and every obligation is consumed, no producer
    can create a late message and the assembler can seal this generation.
    """

    KINDS = frozenset({"chunk", "page", "assembly"})

    def __init__(self, generation: str) -> None:
        self.generation = str(generation)
        self._lock = threading.Lock()
        self._admission_open = True
        self._counts = {kind: 0 for kind in self.KINDS}
        self._sealed = False

    def reserve(self, kind: str, generation: str) -> None:
        if kind not in self.KINDS or generation != self.generation:
            raise ValueError("invalid document fence obligation or generation")
        with self._lock:
            if self._sealed or (
                not self._admission_open and not any(self._counts.values())
            ):
                raise RuntimeError("document producer admission is closed")
            self._counts[kind] += 1

    def complete(self, kind: str, generation: str) -> None:
        if kind not in self.KINDS or generation != self.generation:
            raise ValueError("invalid document fence obligation or generation")
        with self._lock:
            if self._counts[kind] <= 0:
                raise RuntimeError("document fence obligation underflow")
            self._counts[kind] -= 1

    def close_admission(self) -> None:
        with self._lock:
            self._admission_open = False

    def ready(self) -> bool:
        with self._lock:
            return (
                not self._admission_open
                and not self._sealed
                and not any(self._counts.values())
            )

    def seal(self) -> bool:
        with self._lock:
            if (
                self._admission_open
                or self._sealed
                or any(self._counts.values())
            ):
                return False
            self._sealed = True
            return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "generation": self.generation,
                "admission_open": self._admission_open,
                "sealed": self._sealed,
                "counts": dict(self._counts),
            }
