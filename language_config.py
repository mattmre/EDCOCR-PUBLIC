"""backward-compat shim: language_config has moved to ocr_local.config.language_config.

Importing this module redirects to the canonical location.  All existing
import paths (``import language_config``, ``from language_config import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.config.language_config")
