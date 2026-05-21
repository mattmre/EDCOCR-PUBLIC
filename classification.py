"""backward-compat shim: classification has moved to ocr_local.features.classification.

Importing this module redirects to the canonical location.  All existing
import paths (``import classification``, ``from classification import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.features.classification")
