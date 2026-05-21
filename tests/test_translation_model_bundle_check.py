"""Tests for CT2 model-bundle prerequisite validation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import check_translation_model_bundle as checker


def _write_valid_bundle(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"tiny deterministic fixture")
    digest = "a" * 64
    (model_dir / "provenance.json").write_text(
        json.dumps(
            {
                "weights_sha256": digest,
                "license": "CC-BY-4.0",
                "runtime_version": "4.6.1",
                "slsa_provenance_uri": "https://models.example/opus/slsa.intoto.jsonl",
                "intoto_attestation_sha256": digest,
                "sbom_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_missing_model_dir_fails(tmp_path):
    assert checker.main(["--model-dir", str(tmp_path / "missing")]) == 1


def test_missing_enforced_provenance_fails(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"weights")

    assert checker.main(["--model-dir", str(model_dir)]) == 1


def test_not_loaded_weights_digest_fails(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    digest = "b" * 64
    (model_dir / "model.bin").write_bytes(b"weights")
    (model_dir / "provenance.json").write_text(
        json.dumps(
            {
                "weights_sha256": "not_loaded",
                "license": "CC-BY-4.0",
                "runtime_version": "4.6.1",
                "slsa_provenance_uri": "https://models.example/opus/slsa.intoto.jsonl",
                "intoto_attestation_sha256": digest,
                "sbom_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert checker.main(["--model-dir", str(model_dir)]) == 1


def test_valid_bundle_shape_passes(tmp_path):
    model_dir = tmp_path / "model"
    _write_valid_bundle(model_dir)

    assert checker.main(["--model-dir", str(model_dir)]) == 0
