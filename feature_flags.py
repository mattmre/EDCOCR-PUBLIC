"""backward-compat shim: feature_flags has moved to ocr_local.config.feature_flags.

Importing this module redirects to the canonical location.  All existing
import paths (``import feature_flags``, ``from feature_flags import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.config.feature_flags")
