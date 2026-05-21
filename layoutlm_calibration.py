"""backward-compat shim: layoutlm_calibration has moved to ocr_local.ml.layoutlm_calibration.

Importing this module redirects to the canonical location.  All existing
import paths (``import layoutlm_calibration``, ``from layoutlm_calibration import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.ml.layoutlm_calibration")
