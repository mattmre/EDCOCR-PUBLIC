"""backward-compat shim: gpu_optimization has moved to ocr_local.infra.gpu_optimization.

Importing this module redirects to the canonical location.  All existing
import paths (``import gpu_optimization``, ``from gpu_optimization import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.gpu_optimization")
