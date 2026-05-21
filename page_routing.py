"""backward-compat shim: page_routing has moved to ocr_local.infra.page_routing.

Importing this module redirects to the canonical location.  All existing
import paths (``import page_routing``, ``from page_routing import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.infra.page_routing")
