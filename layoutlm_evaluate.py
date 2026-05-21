"""backward-compat shim: layoutlm_evaluate has moved to ocr_local.ml.layoutlm_evaluate.

Importing this module redirects to the canonical location.  All existing
import paths (``import layoutlm_evaluate``, ``from layoutlm_evaluate import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.ml.layoutlm_evaluate")
