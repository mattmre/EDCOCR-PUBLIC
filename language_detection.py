"""backward-compat shim: language_detection has moved to ocr_local.features.language_detection.

Importing this module redirects to the canonical location.  All existing
import paths (``import language_detection``, ``from language_detection import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.features.language_detection")
