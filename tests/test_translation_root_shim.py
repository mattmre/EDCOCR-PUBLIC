"""Tests for the root-level ``translation`` shim module -- Plan B M1-PR5.

The shim follows the  Phase 2 sys.modules-replacement pattern,
preserving access to private/underscore-prefixed names that other
re-export patterns drop.
"""
from __future__ import annotations

import importlib
import sys


def _reload_shim():
    """Force a fresh import of the root ``translation`` module."""
    sys.modules.pop("translation", None)
    return importlib.import_module("translation")


def test_import_translation_resolves():
    mod = _reload_shim()
    assert mod is not None


def test_translation_module_is_ocr_local_translation():
    mod = _reload_shim()
    # The shim replaces sys.modules[__name__] with ocr_local.translation,
    # so __name__ on the resolved module should reference the canonical
    # location.
    assert mod.__name__ == "ocr_local.translation" or mod.__package__ == "ocr_local"


def test_shim_exposes_translate_document():
    """translation.api.translate_document is accessible via the shim."""
    _reload_shim()
    from translation import api  # type: ignore[import-not-found]

    assert callable(api.translate_document)


def test_shim_exposes_engine_registry():
    """translation.ENGINE_REGISTRY is accessible via the shim."""
    mod = _reload_shim()
    assert hasattr(mod, "ENGINE_REGISTRY")
    # passthrough is always registered.
    assert "passthrough" in mod.ENGINE_REGISTRY
