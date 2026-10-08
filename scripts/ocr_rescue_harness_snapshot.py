"""Capture and execute the exact rescue verifier and frozen corpus for one run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot_module(path: Path, name: str) -> ModuleType:
    """Load copied Python bytes, never the later working-tree version."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen harness module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_rescue_harness(
    out: Path,
    manifest: Path,
    *,
    scripts: dict[str, Path],
) -> tuple[Path, dict, ModuleType]:
    """Copy truth and scripts before service startup and return frozen verifier.

    The corpus is tiny and copied in full, so a mid-run edit to the checkout
    cannot replace expected text or source pixels. Every case still checks
    source SHA-256 from the copied manifest during replay and validation.
    """
    snapshot = out / "harness-snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    frozen_manifest_dir = snapshot / "corpus"
    shutil.copytree(manifest.parent, frozen_manifest_dir)
    frozen_manifest = frozen_manifest_dir / manifest.name
    if not frozen_manifest.is_file():
        raise FileNotFoundError("frozen rescue manifest is missing")
    scripts_dir = snapshot / "scripts"
    scripts_dir.mkdir()
    captured: dict[str, str] = {}
    for name, source in scripts.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("snapshot script names must be simple Python filenames")
        destination = scripts_dir / name
        shutil.copy2(source, destination)
        captured[name] = _sha256(destination)
    if "validate_ocr_rescue_outputs.py" not in captured:
        raise ValueError("validator source must be captured")
    corpus_files = {
        str(path.relative_to(frozen_manifest_dir)): _sha256(path)
        for path in sorted(frozen_manifest_dir.rglob("*")) if path.is_file()
    }
    packet = {
        "schema_version": "ocr-rescue-frozen-harness-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": _sha256(frozen_manifest),
        "scripts_sha256": captured,
        "corpus_files_sha256": corpus_files,
    }
    (snapshot / "identity.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validator = load_snapshot_module(
        scripts_dir / "validate_ocr_rescue_outputs.py",
        "_ocr_rescue_frozen_validator",
    )
    return frozen_manifest, packet, validator
