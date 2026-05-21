"""Tests for the Plan A E-A-010 language corpus manifest checker."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import check_language_corpus_manifest as checker


def _write_fixture(root: Path) -> tuple[Path, Path]:
    truth = root / "corpus" / "language_truth.json"
    fixture = root / "fixtures" / "sample.txt"
    truth.parent.mkdir(parents=True)
    fixture.parent.mkdir(parents=True)
    truth.write_text(
        json.dumps({"documents": [{"doc_id": "sample", "language": "en"}]}),
        encoding="utf-8",
    )
    fixture.write_text("sample text\n", encoding="utf-8")
    return truth, fixture


def test_manifest_check_passes_when_hashes_match(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()])
    checker.write_manifest(manifest_path, manifest)

    assert checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
        ]
    ) == 0


def test_manifest_check_fails_when_truth_changes(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()])
    checker.write_manifest(manifest_path, manifest)
    truth.write_text(json.dumps({"documents": []}), encoding="utf-8")

    assert checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
        ]
    ) == 1


def test_manifest_update_requires_explicit_approval(tmp_path, monkeypatch):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    args = [
        "--root",
        str(tmp_path),
        "--truth",
        str(truth),
        "--manifest",
        str(manifest_path),
        "--include",
        fixture.relative_to(tmp_path).as_posix(),
        "--update",
    ]

    monkeypatch.delenv("CORPUS_GOLDEN_SET_UPDATE_APPROVED", raising=False)
    assert checker.main(args) == 1
    assert not manifest_path.exists()

    monkeypatch.setenv("CORPUS_GOLDEN_SET_UPDATE_APPROVED", "true")
    assert checker.main(args) == 0
    assert manifest_path.exists()


def test_manifest_check_fails_when_fixture_changes(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()])
    checker.write_manifest(manifest_path, manifest)
    fixture.write_text("changed text\n", encoding="utf-8")

    assert checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
        ]
    ) == 1


def test_full_gate_fails_when_truth_has_too_few_documents(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()])
    checker.write_manifest(manifest_path, manifest)

    assert checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
            "--require-full-gate",
        ]
    ) == 1


def test_full_gate_passes_with_required_document_count(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps({"documents": [{"doc_id": str(i), "language": "en"} for i in range(50)]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()])
    checker.write_manifest(manifest_path, manifest)

    assert checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
            "--require-full-gate",
        ]
    ) == 0
