"""Engine registry -- import-side-effect-free.

The registry is a plain dict keyed by ``EngineCapability.id`` and is
populated by the ``@register_engine`` class decorator.  Heavy engines
backed by ``ctranslate2`` are imported lazily so that environments
without that runtime (notably the SDK CI lane) can still import the
package and run the contract test.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_local.translation.engines.base import TranslationEngine

ENGINE_REGISTRY: dict[str, type["TranslationEngine"]] = {}


def register_engine(cls: type["TranslationEngine"]) -> type["TranslationEngine"]:
    """Class decorator -- registers ``cls`` under ``cls.capability.id``."""

    ENGINE_REGISTRY[cls.capability.id] = cls
    return cls


def get_engine(engine_id: str) -> type["TranslationEngine"]:
    """Look up a registered engine class by id.

    Raises ``KeyError`` with the list of known engine ids when the
    requested id is not registered.
    """

    if engine_id not in ENGINE_REGISTRY:
        raise KeyError(
            f"Unknown engine: {engine_id!r}. Registered: {list(ENGINE_REGISTRY)}"
        )
    return ENGINE_REGISTRY[engine_id]


def iter_engines():
    """Iterate ``(engine_id, engine_class)`` pairs in registration order."""

    return ENGINE_REGISTRY.items()


def _lazy_register_ct2_engines() -> None:
    """Register CT2-backed engines only when ``ctranslate2`` is installed.

    The passthrough reference engine is always registered so the contract
    test has at least one real adapter to exercise.
    """

    if importlib.util.find_spec("ctranslate2") is not None:
        # The CT2 modules are imported for their registration side
        # effects; they don't expose anything we need to bind locally.
        importlib.import_module("ocr_local.translation.engines.local_ct2_opus")
        importlib.import_module("ocr_local.translation.engines.local_ct2_nllb")
        importlib.import_module("ocr_local.translation.engines.local_ct2_madlad")

    # Always register the passthrough reference adapter.
    importlib.import_module("ocr_local.translation.engines.passthrough")


# Register on import so that ``ENGINE_REGISTRY`` is populated for any
# caller that imports ``ocr_local.translation``.
_lazy_register_ct2_engines()
