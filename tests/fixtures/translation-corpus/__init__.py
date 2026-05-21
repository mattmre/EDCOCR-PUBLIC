"""Translation test corpus loader.

Reads FLORES-200 derived TSV pairs from ``flores200/<src>_<tgt>.tsv``.
Returns ``[]`` for unknown directions so callers can degrade gracefully
when a particular language pair has not been seeded.
"""

from __future__ import annotations

import csv
import os

_CORPUS_DIR = os.path.dirname(__file__)


def load_flores_pairs(src: str, tgt: str) -> list[tuple[str, str]]:
    """Load FLORES-200 sentence pairs for a language direction.

    Returns a list of ``(source_text, target_text)`` tuples; rows with
    fewer than two columns are skipped silently.
    """

    path = os.path.join(_CORPUS_DIR, "flores200", f"{src}_{tgt}.tsv")
    if not os.path.exists(path):
        return []
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0], row[1]))
    return pairs
