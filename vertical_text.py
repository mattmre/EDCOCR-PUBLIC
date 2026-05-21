"""backward-compat shim: vertical_text has moved to ocr_local.features.vertical_text.

Importing this module redirects to the canonical location.  All existing
import paths (``import vertical_text``, ``from vertical_text import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module(
    "ocr_local.features.vertical_text"
)
