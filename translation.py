"""backward-compat shim: translation has moved to ocr_local.translation.

Importing this module redirects to the canonical location.  All existing
import paths (``import translation``, ``from translation import X``)
continue to work unchanged, including access to private symbols.

 Phase 2 sys.modules replacement pattern -- preserves underscore-
prefixed symbols by replacing the whole module entry rather than
re-exporting via ``__all__`` or ``from X import Y``.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.translation")
