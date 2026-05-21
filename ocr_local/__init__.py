"""ocr_local -- organised namespace package for EDCOCR modules.

This package provides the canonical import namespace for the EDCOCR
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

The namespace was introduced via a meta-path finder without moving any
files; modules were then physically migrated into their canonical sub-package.
The meta-path finder is now inert; ``install()`` is a no-op kept only for
backward compatibility.  Root-level shim modules may be removed in a future
release after a deprecation cycle.
"""

from ocr_local._compat_finder import install as _install_compat_finder

# Retained for backward compatibility; this is a no-op.
_install_compat_finder()

__version__ = "1.0.0-compat"
