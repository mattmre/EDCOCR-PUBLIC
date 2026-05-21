"""Infra modules -- engine selection, caching, routing, metrics.

Each infrastructure module is a first-class sub-package module and is
imported through the standard Python machinery::

    from ocr_local.infra import engine_selection
    import ocr_local.infra.page_cache as cache

 Phase 2 is complete: every infra module has been physically
migrated into this package, so no PEP-562 ``__getattr__`` lazy-loader is
required.  ``_INFRA_MODULES`` is retained as an empty tuple so that
legacy introspection (``from ocr_local.infra import _INFRA_MODULES``)
continues to resolve.
"""

from __future__ import annotations

# Retained as a stable public symbol for legacy introspection and tests.
_INFRA_MODULES: tuple[str, ...] = ()
