"""backward-compat shim: version has moved to ocr_local.config.version.

Importing this module redirects to the canonical location.  All existing
import paths (``import version``, ``from version import __version__``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.config.version")
