"""Tests for the bootstrap translation golden corpus scoring harness."""

from __future__ import annotations

import json

from scripts import score_translation_golden_corpus as scorer


def test_reference_scoring_passes_for_matching_reference(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "s1",
                        "source_lang": "en",
                        "target_lang": "fr",
                        "source": "Hello.",
                        "reference": "Bonjour.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert scorer.main(["--manifest", str(manifest)]) == 0


def test_reference_scoring_requires_segments(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"segments": []}), encoding="utf-8")

    assert scorer.main(["--manifest", str(manifest)]) == 1
