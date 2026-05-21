"""backward-compat shim: dpi_escalation has moved to ocr_local.features.dpi_escalation.

Importing this module redirects to the canonical location.  All existing
import paths (``import dpi_escalation``, ``from dpi_escalation import X``)
continue to work unchanged, including access to private symbols.
"""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("ocr_local.features.dpi_escalation")
