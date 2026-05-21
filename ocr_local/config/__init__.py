"""Config modules -- environment schema, feature flags, language registry.

Each config module is a first-class sub-package module and is imported
through the standard Python machinery::

    from ocr_local.config import language_config
    import ocr_local.config.version as version_mod

 Phase 2 is complete: every config module has been physically
migrated into this package, so no PEP-562 ``__getattr__`` lazy-loader is
required.  ``_CONFIG_MODULES`` is retained as an empty tuple so that
legacy introspection (``from ocr_local.config import _CONFIG_MODULES``)
continues to resolve.
"""

from __future__ import annotations

# Retained as a stable public symbol for legacy introspection and tests.
_CONFIG_MODULES: tuple[str, ...] = ()
