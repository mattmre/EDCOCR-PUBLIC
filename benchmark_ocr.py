"""backward-compat shim: benchmark_ocr has moved to ocr_local.infra.benchmark_ocr.

Importing this module redirects to the canonical location.  All existing
import paths (``import benchmark_ocr``, ``from benchmark_ocr import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.benchmark_ocr")
