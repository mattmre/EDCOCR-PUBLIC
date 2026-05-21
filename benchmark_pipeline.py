"""backward-compat shim: benchmark_pipeline has moved to ocr_local.infra.benchmark_pipeline.

Importing this module redirects to the canonical location.  All existing
import paths (``import benchmark_pipeline``, ``from benchmark_pipeline import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.benchmark_pipeline")
