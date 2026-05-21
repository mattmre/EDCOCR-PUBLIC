"""Inert  Phase 2 compatibility surface.

Background
----------
Phase 1 of  introduced a meta-path finder that mapped dotted names
like ``ocr_local.features.ner`` to root-level modules without moving any
files.  Phase 2 has now physically migrated every sub-module into its
canonical ``ocr_local/<subpkg>/`` location, so the finder and loader
classes are no longer required -- Python's normal import machinery
resolves each name directly.

This module is kept as a stable import surface for existing call sites
and tests, but all meta-path behavior has been removed:

- ``_SUBPACKAGE_MODULES`` remains as the historical dict keyed by sub-
  package dotted name, with every value frozen to an empty frozenset.
- ``install()`` is a no-op retained for backward compatibility with
  ``ocr_local/__init__.py``; re-running it is safe.

If a future refactor reintroduces Phase 1-style shims, reinstating the
finder is straightforward, but by design Phase 2 leaves this layer
inert.
"""

from __future__ import annotations

# Historical map of compat sub-package name -> allowed root module names.
# After Phase 2 every sub-package owns its own files, so the frozensets
# are empty.  The dict is retained so existing tests and callers that
# import ``_SUBPACKAGE_MODULES`` continue to work.
_SUBPACKAGE_MODULES: dict[str, frozenset[str]] = {
    "ocr_local.features": frozenset(),
    "ocr_local.ml": frozenset(),
    "ocr_local.infra": frozenset(),
    "ocr_local.config": frozenset(),
}


def install() -> None:
    """No-op retained for backward compatibility.

    Prior to  Phase 2 this registered a ``MetaPathFinder`` on
    ``sys.meta_path``.  Phase 2 migrated every module into its canonical
    sub-package, so no meta-path hook is needed; the function is kept so
    that ``ocr_local/__init__.py`` (and any third-party callers) can
    continue to invoke it without change.
    """
    return None
