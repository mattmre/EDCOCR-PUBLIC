"""ocr_local -- organised namespace package for OCR_LOCAL modules.

This package provides the canonical import namespace for the OCR_LOCAL
pipeline modules.  All existing root-level import paths (e.g.
``from ner import ...``) remain valid via thin shim modules; this
package exposes the structured sub-package layout that long-term callers
should target.

Sub-packages
------------
features/   Opt-in OCR feature modules (NER, classification, etc.)
ml/         Machine-learning modules (LayoutLMv3, calibration, etc.)
infra/      Infrastructure and runtime modules (engine selection, caching, etc.)
config/     Configuration and language registry modules

maintenance review Phase 1 introduced this namespace via a meta-path finder without
moving any files.  maintenance review Phase 2 physically migrated every module into
its canonical sub-package; the meta-path finder is now inert and
``install()`` is a no-op, kept only for backward compatibility.  Phase 3
(future) may remove the root-level shim modules after a deprecation
cycle.
"""

from ocr_local._compat_finder import install as _install_compat_finder

# Retained for backward compatibility; this is a no-op after maintenance review Ph2.
_install_compat_finder()

__version__ = "1.0.0-compat"
