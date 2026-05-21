"""Tests for Qwen/COMETKiwi golden corpus prerequisite validation."""

from __future__ import annotations

import json
import os

from scripts import check_translation_golden_corpus as checker
from scripts import generate_translation_golden_corpus as generator


def test_missing_manifest_fails(tmp_path):
    assert checker.main(["--corpus-dir", str(tmp_path), "--min-segments", "1"]) == 1


def test_manifest_requires_quality_evidence_keys(tmp_path):
    corpus_dir = tmp_path / "golden"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "language_pairs": ["en-fr"], "segments": []}),
        encoding="utf-8",
    )

    assert checker.main(["--corpus-dir", str(corpus_dir), "--min-segments", "1"]) == 1


def test_manifest_with_minimal_valid_shape_passes(tmp_path):
    corpus_dir = tmp_path / "golden"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "corpus_id": "test",
                "language_pairs": ["en-fr"],
                "license_attribution": "test fixture",
                "segments": [
                    {
                        "id": "en-fr-0001",
                        "source_lang": "en",
                        "target_lang": "fr",
                        "source": "Hello world.",
                        "reference": "Bonjour le monde.",
                        "domain": "test",
                        "source_fixture": "unit-test",
                        "source_row": 1,
                        "license": "test",
                        "sha256": "",
                    }
                ],
                "cometkiwi_baselines": {"en-fr": {"threshold": 0.7}},
                "qwen_comparison_report": {"status": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["segments"][0]["sha256"] = checker._segment_hash(payload["segments"][0])
    (corpus_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (corpus_dir / "manifest.sha256").write_text(
        checker._file_hash(corpus_dir / "manifest.json") + "\n",
        encoding="utf-8",
    )

    assert checker.main(["--corpus-dir", str(corpus_dir), "--min-segments", "1"]) == 0


def test_generated_30_segment_corpus_fails_200_gate(tmp_path):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "30"])
    os.environ[checker.APPROVAL_ENV] = "true"
    try:
        assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 1
    finally:
        os.environ.pop(checker.APPROVAL_ENV, None)


def test_generated_200_segment_corpus_passes(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "200"])
    monkeypatch.setenv(checker.APPROVAL_ENV, "true")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 0
    monkeypatch.delenv(checker.APPROVAL_ENV)
    assert checker.main(["--corpus-dir", str(corpus_dir), "--min-segments", "200"]) == 0


def test_manifest_hash_drift_fails(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "200"])
    monkeypatch.setenv(checker.APPROVAL_ENV, "true")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 0
    monkeypatch.delenv(checker.APPROVAL_ENV)
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["segments"][0]["reference"] = "drifted"
    (corpus_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--min-segments", "200"]) == 1


def test_segment_hash_drift_after_first_25_fails(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "200"])
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["segments"][26]["reference"] = "drifted after first 25"
    payload["segments"][26]["sha256"] = checker._segment_hash(payload["segments"][26])
    (corpus_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(checker.APPROVAL_ENV, "true")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 0
    monkeypatch.delenv(checker.APPROVAL_ENV)
    payload["segments"][26]["reference"] = "unapproved drift"
    (corpus_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--min-segments", "200"]) == 1


def test_missing_license_or_source_metadata_fails(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "200"])
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["segments"][0].pop("license")
    (corpus_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(checker.APPROVAL_ENV, "true")
    assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 1


def test_manifest_hash_update_requires_approval(tmp_path):
    corpus_dir = tmp_path / "golden"
    generator.main(["--corpus-dir", str(corpus_dir), "--segments", "200"])
    assert checker.main(["--corpus-dir", str(corpus_dir), "--update-manifest-hash"]) == 1
