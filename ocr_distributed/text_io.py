"""Bounded text-artifact reads shared by coordinator workers."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MAX_TEXT_BYTES = 16 * 1024 * 1024


def read_text_bounded(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """Read a text artifact only when it fits the configured memory budget.

    OCR text is normally small, but an untrusted or corrupted artifact must
    not turn a worker into an unbounded ``read()`` memory sink.  Oversized
    input fails closed so the original artifact remains available for audit.
    """
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    source = Path(path)
    try:
        if source.stat().st_size > max_bytes:
            raise ValueError(f"text artifact exceeds the {max_bytes}-byte safety limit")
    except OSError:
        # Let the bounded open below return the authoritative filesystem
        # error, rather than converting a missing path into a false success.
        pass

    with source.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"text artifact exceeds the {max_bytes}-byte safety limit")
    return raw.decode(encoding, errors=errors)


__all__ = ["DEFAULT_MAX_TEXT_BYTES", "read_text_bounded"]
