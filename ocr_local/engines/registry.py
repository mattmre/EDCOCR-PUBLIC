"""Engine registry: register / discover pluggable OCR engines by id.

Thread-safe, lazy-instantiating registry mirroring the entry-point discovery
shape of :func:`ocr_local.enrichment_bus.discover_entry_point_providers`.
Engine factories must build cheaply (model loading is deferred to
``OCREngine.get_handle``), so instantiating one to read its capability is safe.
"""
from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import Callable

from ocr_local.engines.base import OCREngine
from ocr_local.engines.capability import EngineCapability

logger = logging.getLogger(__name__)

EngineFactory = Callable[[], OCREngine]


class EngineRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, EngineFactory] = {}
        self._instances: dict[str, OCREngine] = {}
        self._lock = threading.RLock()

    def register(
        self, engine_id: str, factory: EngineFactory, *, replace: bool = False
    ) -> None:
        if not engine_id:
            raise ValueError("engine_id is required")
        with self._lock:
            if engine_id in self._factories and not replace:
                raise ValueError(f"engine {engine_id!r} already registered")
            self._factories[engine_id] = factory
            self._instances.pop(engine_id, None)

    def unregister(self, engine_id: str) -> None:
        with self._lock:
            self._factories.pop(engine_id, None)
            self._instances.pop(engine_id, None)

    def is_registered(self, engine_id: str) -> bool:
        with self._lock:
            return engine_id in self._factories

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def get(self, engine_id: str) -> OCREngine:
        """Return the (memoized) engine instance for ``engine_id``."""
        with self._lock:
            if engine_id not in self._factories:
                raise KeyError(f"engine {engine_id!r} is not registered")
            instance = self._instances.get(engine_id)
            if instance is None:
                instance = self._factories[engine_id]()
                self._instances[engine_id] = instance
            return instance

    def capabilities(self) -> list[EngineCapability]:
        """Capability records for all registered engines (skips broken ones)."""
        caps: list[EngineCapability] = []
        for engine_id in self.list_ids():
            try:
                caps.append(self.get(engine_id).capability)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("engine %s capability unavailable: %s", engine_id, exc)
        return caps

    def available_ids(self) -> list[str]:
        """Subset of registered engines whose runtime deps/models are loadable."""
        out: list[str] = []
        for engine_id in self.list_ids():
            try:
                if self.get(engine_id).is_available():
                    out.append(engine_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("engine %s availability check failed: %s", engine_id, exc)
        return out

    def discover_entry_points(self, group: str = "ocr_local.engines") -> list[str]:
        """Register engines published via the ``ocr_local.engines`` entry-point group."""
        registered: list[str] = []
        entry_points = importlib.metadata.entry_points()
        for entry_point in entry_points.select(group=group):
            factory = entry_point.load()
            self.register(entry_point.name, factory, replace=True)
            registered.append(entry_point.name)
        return registered


# Process-wide default registry.  Adapters (Phase 1) register here at import.
DEFAULT_REGISTRY = EngineRegistry()


def get_default_registry() -> EngineRegistry:
    return DEFAULT_REGISTRY
