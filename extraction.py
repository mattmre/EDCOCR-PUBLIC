"""backward-compat shim: extraction has moved to ocr_local.features.extraction.

Importing this module redirects to the canonical location.  All existing
import paths (``import extraction``, ``from extraction import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.features.extraction")
