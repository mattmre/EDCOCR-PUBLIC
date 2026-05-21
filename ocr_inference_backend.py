"""backward-compat shim: ocr_inference_backend has moved to ocr_local.infra.ocr_inference_backend.

Importing this module redirects to the canonical location.  All existing
import paths (``import ocr_inference_backend``, ``from ocr_inference_backend import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.ocr_inference_backend")
