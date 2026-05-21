"""backward-compat shim: engine_selection has moved to ocr_local.infra.engine_selection.

Importing this module redirects to the canonical location.  All existing
import paths (``import engine_selection``, ``from engine_selection import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.engine_selection")
