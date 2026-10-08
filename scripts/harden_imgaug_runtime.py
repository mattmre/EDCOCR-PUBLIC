#!/usr/bin/env python3
"""Disable imgaug 0.4.0's unused, unsafe deprecated BackgroundAugmenter.

CVE-2026-31235 has no upstream fixed release. Paddle inference uses synchronous
augmentation; this image deliberately does not support background training
augmentation. Keep the upstream advisory in the inventory as mitigated, not
scanner-clean. Unknown upstream bytes fail the build for manual review.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

ORIGINAL_SHA256 = "3ea4d6b4eb6df208e2425978ff41c725283d64ea29eb4d4b3584c44daa290e73"
STUB = '''class BackgroundAugmenter(object):
    """Disabled in OCR runtime: CVE-2026-31235, no background training support."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "imgaug BackgroundAugmenter is disabled in OCR runtime (CVE-2026-31235)"
        )
'''


def harden_source(source: bytes, *, expected_sha256: str = ORIGINAL_SHA256) -> bytes:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise RuntimeError(
            "unreviewed imgaug multicore source; refusing automatic patch"
        )
    text = source.decode("utf-8")
    nodes = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.ClassDef) and node.name == "BackgroundAugmenter"
    ]
    if len(nodes) != 1:
        raise RuntimeError("expected exactly one BackgroundAugmenter class")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    replacement = (
        "".join(lines[: node.lineno - 1]) + STUB + "".join(lines[node.end_lineno :])
    )
    compile(replacement, "imgaug/multicore.py", "exec")
    return replacement.encode("utf-8")


def main() -> None:
    if importlib.metadata.version("imgaug") != "0.4.0":
        raise RuntimeError(
            "unreviewed imgaug version; revisit upstream advisory and patch"
        )
    spec = importlib.util.find_spec("imgaug")
    if spec is None or spec.origin is None:
        raise RuntimeError("imgaug package is missing")
    path = Path(spec.origin).parent / "multicore.py"
    patched = harden_source(path.read_bytes())
    path.write_bytes(patched)
    # Never allow an old compiled module to bypass the patched source.
    for cached in (path.parent / "__pycache__").glob("multicore.*.pyc"):
        cached.unlink()
    print(
        "IMGAUG_BACKGROUND_AUGMENTER_DISABLED sha256="
        + hashlib.sha256(patched).hexdigest()
    )


if __name__ == "__main__":
    main()
