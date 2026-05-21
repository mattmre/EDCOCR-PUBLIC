"""Feature modules -- opt-in OCR pipeline capabilities.

Each feature module is a first-class sub-package module and is imported
through the standard Python machinery::

    from ocr_local.features import ner           # attribute access
    import ocr_local.features.ner as ner_mod     # direct import
    from ocr_local.features.ner import NERExtractor

Root-level paths such as ``import ner`` continue to work through thin
shim modules at the repository root.

 Phase 2 is complete: every feature module has been physically
migrated into this package, so no PEP-562 ``__getattr__`` lazy-loader is
required.  ``_FEATURE_MODULES`` is retained as an empty tuple so that
legacy introspection (``from ocr_local.features import _FEATURE_MODULES``)
continues to resolve.
"""

from __future__ import annotations

# Retained as a stable public symbol for legacy introspection and tests.
# After  Phase 2 every feature module lives in this package and is
# resolved by the standard import machinery, so this tuple is empty.
_FEATURE_MODULES: tuple[str, ...] = ()
