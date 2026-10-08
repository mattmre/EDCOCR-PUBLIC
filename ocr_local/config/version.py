"""OCR_LOCAL version information.

Public releases use strict ``MAJOR.MINOR.PATCH``.
"""

from __future__ import annotations

import re

__version__ = "4.2.0-rc.1"
__release_stage__ = "candidate"
__public_version_frozen__ = False

_m = re.match(r"^(\d+)\.(\d+)\.(\d+)", __version__)
if not _m:
    raise RuntimeError(f"unparseable __version__: {__version__!r}")
__version_info__ = tuple(int(x) for x in _m.groups())
