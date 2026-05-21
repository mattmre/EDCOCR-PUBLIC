"""backward-compat shim: noise_profiling has moved to ocr_local.features.noise_profiling.

Importing this module redirects to the canonical location.  All existing
import paths (``import noise_profiling``, ``from noise_profiling import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

# Replace this partially-initialised shim entry in sys.modules with the real
# module so that subsequent attribute lookups resolve against the actual
# implementation.
_sys.modules[__name__] = _importlib.import_module("ocr_local.features.noise_profiling")
