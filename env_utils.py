"""backward-compat shim: env_utils has moved to ocr_local.config.env_utils.

Importing this module redirects to the canonical location.  All existing
import paths (``import env_utils``, ``from env_utils import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.config.env_utils")
