"""backward-compat shim: adaptive_batch has moved to ocr_local.infra.adaptive_batch.

Importing this module redirects to the canonical location.  All existing
import paths (``import adaptive_batch``, ``from adaptive_batch import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.adaptive_batch")
