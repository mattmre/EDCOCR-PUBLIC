"""backward-compat shim: layoutlm_data has moved to ocr_local.ml.layoutlm_data.

Importing this module redirects to the canonical location.  All existing
import paths (``import layoutlm_data``, ``from layoutlm_data import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.ml.layoutlm_data")
