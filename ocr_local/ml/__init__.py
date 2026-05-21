"""ML modules -- LayoutLMv3 fine-tuning, evaluation, and calibration.

Each ML module is a first-class sub-package module and is imported
through the standard Python machinery::

    from ocr_local.ml import layoutlm_labels
    import ocr_local.ml.layoutlm_finetune as ft

 Phase 2 is complete: every ML module has been physically
migrated into this package, so no PEP-562 ``__getattr__`` lazy-loader is
required.  ``_ML_MODULES`` is retained as an empty tuple so that legacy
introspection (``from ocr_local.ml import _ML_MODULES``) continues to
resolve.
"""

from __future__ import annotations

# Retained as a stable public symbol for legacy introspection and tests.
_ML_MODULES: tuple[str, ...] = ()
