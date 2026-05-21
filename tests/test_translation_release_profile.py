"""Tests for Plan B translation provenance release-profile enforcement."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from scripts import validate_translation_release_profile as validator


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_profile_fails_when_translation_enabled_without_provenance(tmp_path):
    profile = tmp_path / "values-staging.yaml"
    _write(
        profile,
        """
        coordinator:
          env:
            ENABLE_TRANSLATION: "true"
        """,
    )

    assert validator.main(["--profile-file", str(profile), "--model-dir", str(tmp_path / "none")]) == 1


def test_profile_passes_when_translation_enforces_provenance(tmp_path):
    profile = tmp_path / "values-staging.yaml"
    _write(
        profile,
        """
        coordinator:
          env:
            ENABLE_TRANSLATION: "true"
            OCR_TRANSLATION_ENFORCE_PROVENANCE: "true"
        """,
    )

    assert validator.main(["--profile-file", str(profile), "--model-dir", str(tmp_path / "none")]) == 0


def test_profile_ignores_disabled_translation(tmp_path):
    profile = tmp_path / "values-production.yaml"
    _write(
        profile,
        """
        coordinator:
          env:
            ENABLE_TRANSLATION: "false"
        """,
    )

    assert validator.main(["--profile-file", str(profile), "--model-dir", str(tmp_path / "none")]) == 0


def test_model_dir_fails_invalid_provenance_json(tmp_path):
    profile = tmp_path / "values-staging.yaml"
    model = tmp_path / "models" / "opus"
    _write(
        profile,
        """
        coordinator:
          env:
            ENABLE_TRANSLATION: "true"
            OCR_TRANSLATION_ENFORCE_PROVENANCE: "true"
        """,
    )
    _write(
        model / "provenance.json",
        json.dumps(
            {
                "weights_sha256": "not_loaded",
                "license": "CC-BY-4.0",
                "runtime_version": "4.6.1",
            }
        ),
    )

    assert validator.main(["--profile-file", str(profile), "--model-dir", str(tmp_path / "models")]) == 1


def test_model_dir_passes_valid_provenance_json(tmp_path):
    profile = tmp_path / "values-staging.yaml"
    model = tmp_path / "models" / "opus"
    digest = "a" * 64
    _write(
        profile,
        """
        coordinator:
          env:
            ENABLE_TRANSLATION: "true"
            OCR_TRANSLATION_ENFORCE_PROVENANCE: "true"
        """,
    )
    _write(
        model / "provenance.json",
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
    )

    assert validator.main(["--profile-file", str(profile), "--model-dir", str(tmp_path / "models")]) == 0
