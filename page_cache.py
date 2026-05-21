"""backward-compat shim: page_cache has moved to ocr_local.infra.page_cache.

Importing this module redirects to the canonical location.  All existing
import paths (``import page_cache``, ``from page_cache import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.page_cache")
