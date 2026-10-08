"""Tests for the Plan A E-A-010 language corpus manifest checker."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import check_language_corpus_manifest as checker


def test_cli_bootstraps_repo_import_from_outside_checkout(tmp_path):
    script = Path(checker.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "--require-full-gate" in result.stdout


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
    manifest = checker.build_manifest(
        tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()]
    )
    checker.write_manifest(manifest_path, manifest)

    assert (
        checker.main(
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
        )
        == 0
    )


def test_manifest_check_fails_when_truth_changes(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(
        tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()]
    )
    checker.write_manifest(manifest_path, manifest)
    truth.write_text(json.dumps({"documents": []}), encoding="utf-8")

    assert (
        checker.main(
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
        )
        == 1
    )


def test_manifest_update_requires_approval_and_explicit_full_gate(
    tmp_path, monkeypatch, capsys
):
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
    assert "CORPUS_GOLDEN_SET_UPDATE_APPROVED=true" in capsys.readouterr().err

    monkeypatch.setenv("CORPUS_GOLDEN_SET_UPDATE_APPROVED", "true")
    assert checker.main(args) == 1
    assert not manifest_path.exists()
    assert "--update requires --require-full-gate" in capsys.readouterr().err


def test_full_gate_update_does_not_write_without_full_corpus(
    tmp_path, monkeypatch, capsys
):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    monkeypatch.setenv("CORPUS_GOLDEN_SET_UPDATE_APPROVED", "true")

    result = checker.main(
        [
            "--root",
            str(tmp_path),
            "--truth",
            str(truth),
            "--manifest",
            str(manifest_path),
            "--include",
            fixture.relative_to(tmp_path).as_posix(),
            "--update",
            "--require-full-gate",
        ]
    )

    assert result == 1
    assert not manifest_path.exists()
    assert "language corpus full-gate check failed" in capsys.readouterr().err


def test_manifest_check_fails_when_fixture_changes(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(
        tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()]
    )
    checker.write_manifest(manifest_path, manifest)
    fixture.write_text("changed text\n", encoding="utf-8")

    assert (
        checker.main(
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
        )
        == 1
    )


def test_full_gate_fails_when_truth_has_too_few_documents(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(
        tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()]
    )
    checker.write_manifest(manifest_path, manifest)

    assert (
        checker.main(
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
        )
        == 1
    )


def test_full_gate_reports_non_object_truth_root(tmp_path):
    truth = tmp_path / "language_truth.json"
    truth.write_text(json.dumps([]), encoding="utf-8")

    assert checker._full_gate_problems(tmp_path, truth) == [
        "language truth root must be an object"
    ]


def test_full_gate_reports_non_object_document_row(tmp_path):
    truth = tmp_path / "language_truth.json"
    truth.write_text(json.dumps({"documents": [None]}), encoding="utf-8")

    problems = checker._full_gate_problems(tmp_path, truth)

    assert "document row 1 is not an object" in problems


def test_full_gate_requires_non_empty_string_document_ids(tmp_path):
    truth = tmp_path / "language_truth.json"
    truth.write_text(json.dumps({"documents": [{"doc_id": 17}]}), encoding="utf-8")

    problems = checker._full_gate_problems(tmp_path, truth)

    assert "document row 1: doc_id must be a non-empty string" in problems


def test_full_gate_reports_non_list_page_languages(tmp_path):
    truth, _fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "malformed-page",
                        "expected_page_languages": [
                            {
                                "primary_language": "en",
                                "languages_present": None,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert (
        "malformed-page: page row 1: languages_present must be a non-empty list"
        in problems
    )


def test_full_gate_rejects_malformed_fixture_metadata_and_page_order(tmp_path):
    truth, _fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "malformed-metadata",
                        "source_fixture": 42,
                        "source_sha256": 42,
                        "expected_document_language": "en",
                        "document_class": "monolingual Latin document",
                        "expected_page_languages": [
                            {
                                "page": 2,
                                "primary_language": "en",
                                "languages_present": ["en"],
                            },
                            {
                                "page": 1,
                                "primary_language": "en",
                                "languages_present": ["en"],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert (
        "malformed-metadata: source_fixture must be a non-empty relative path"
        in problems
    )
    assert "malformed-metadata: source_sha256 is missing or invalid" in problems
    assert "malformed-metadata: page numbers must be in ascending order" in problems


def test_full_gate_does_not_treat_negated_class_labels_as_coverage(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "negative-coverage",
                        "source_fixture": fixture.relative_to(tmp_path).as_posix(),
                        "source_sha256": checker._format_sha256(
                            checker._sha256(fixture)
                        ),
                        "expected_document_language": "en",
                        "document_class": "not degraded and not low-resolution",
                        "expected_page_languages": [
                            {
                                "page": 1,
                                "primary_language": "en",
                                "languages_present": ["en"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert "required E-A-010 coverage missing: degraded/low-resolution scan" in problems

    truth_data = json.loads(truth.read_text(encoding="utf-8"))
    truth_data["documents"][0]["document_class"] = (
        "not degraded but low-resolution scan"
    )
    truth.write_text(json.dumps(truth_data), encoding="utf-8")

    problems = checker._full_gate_problems(tmp_path, truth)

    assert (
        "required E-A-010 coverage missing: degraded/low-resolution scan"
        not in problems
    )


def test_full_gate_rejects_document_count_without_required_coverage(tmp_path, capsys):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {"documents": [{"doc_id": str(i), "language": "en"} for i in range(50)]}
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(
        tmp_path, truth, [fixture.relative_to(tmp_path).as_posix()]
    )
    checker.write_manifest(manifest_path, manifest)

    assert (
        checker.main(
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
        )
        == 1
    )
    assert "required E-A-010 coverage missing" in capsys.readouterr().err


def test_language_aliases_resolve_to_one_canonical_registry_code():
    assert checker._canonical_language("eng") == "en"
    assert checker._canonical_language("zh-Hans") == "ch"
    assert checker._canonical_language("unregistered") is None


def test_full_gate_rejects_unregistered_labels_as_bilingual_coverage(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "forged-transition",
                        "source_fixture": fixture.relative_to(tmp_path).as_posix(),
                        "source_sha256": checker._format_sha256(
                            checker._sha256(fixture)
                        ),
                        "expected_document_language": "zz-one",
                        "document_class": "bilingual page transition",
                        "expected_page_languages": [
                            {
                                "page": 1,
                                "primary_language": "zz-one",
                                "languages_present": ["zz-one"],
                            },
                            {
                                "page": 2,
                                "primary_language": "zz-two",
                                "languages_present": ["zz-two"],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert any(
        "zz-one" in problem and "supported language code" in problem
        for problem in problems
    )
    assert any(
        "zz-two" in problem and "supported language code" in problem
        for problem in problems
    )
    assert (
        "required E-A-010 coverage missing: bilingual/multilingual page transition"
        in problems
    )


def test_full_gate_uses_page_truth_for_rtl_ltr_coverage_and_checks_primary_membership(
    tmp_path,
):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "expected-only-rtl",
                        "source_fixture": fixture.relative_to(tmp_path).as_posix(),
                        "source_sha256": checker._format_sha256(
                            checker._sha256(fixture)
                        ),
                        "expected_document_language": "ar",
                        "document_class": "mixed RTL and LTR document",
                        "expected_page_languages": [
                            {
                                "page": 1,
                                "primary_language": "eng",
                                "languages_present": ["fr"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert (
        "expected-only-rtl: expected_document_language must match at least one "
        "page primary_language"
        in problems
    )
    assert (
        "expected-only-rtl: page row 1: primary_language must appear in "
        "languages_present"
        in problems
    )
    assert "required E-A-010 coverage missing: RTL + LTR mixed document" in problems


def test_full_gate_rejects_duplicate_source_fixture_rows(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    source_hash = checker._format_sha256(checker._sha256(fixture))
    source_fixture = fixture.relative_to(tmp_path).as_posix()

    def document(doc_id, language, document_class, pages, **extra):
        return {
            "doc_id": doc_id,
            "source_fixture": source_fixture,
            "source_sha256": source_hash,
            "expected_document_language": language,
            "document_class": document_class,
            "expected_page_languages": pages,
            **extra,
        }

    one_page_en = [{"page": 1, "primary_language": "en", "languages_present": ["en"]}]
    documents = [
        document(
            f"same-source-en-{index}",
            "en",
            "monolingual Latin document",
            one_page_en,
        )
        for index in range(43)
    ]
    documents.extend(
        [
            document(
                "same-source-fr",
                "fr",
                "monolingual Latin document",
                [{"page": 1, "primary_language": "fr", "languages_present": ["fr"]}],
            ),
            document(
                "same-source-cjk",
                "mixed-cjk",
                "mixed CJK scan",
                [
                    {
                        "page": 1,
                        "primary_language": "mixed-cjk",
                        "languages_present": ["zh", "ja"],
                    }
                ],
            ),
            document(
                "same-source-rtl-ltr",
                "ar",
                "mixed RTL and LTR document",
                [
                    {
                        "page": 1,
                        "primary_language": "ar",
                        "languages_present": ["ar", "en"],
                    }
                ],
            ),
            document(
                "same-source-und",
                "und",
                "short noisy low-confidence document",
                [{"page": 1, "primary_language": "und", "languages_present": ["und"]}],
            ),
            document(
                "same-source-extended",
                "th",
                "extended-tier language lockout case",
                [{"page": 1, "primary_language": "th", "languages_present": ["th"]}],
                extended_tier_lockout=True,
            ),
            document(
                "same-source-transition",
                "en",
                "bilingual page transition",
                [
                    {"page": 1, "primary_language": "en", "languages_present": ["en"]},
                    {"page": 2, "primary_language": "fr", "languages_present": ["fr"]},
                ],
            ),
            document(
                "same-source-degraded",
                "en",
                "degraded low-resolution scan",
                one_page_en,
            ),
        ]
    )
    truth.write_text(json.dumps({"documents": documents}), encoding="utf-8")

    problems = checker._full_gate_problems(tmp_path, truth)

    assert any(
        "same-source-en-1: source_fixture duplicates document same-source-en-0"
        in problem
        for problem in problems
    )
    assert "1 unique source fixtures present; 50 required for E-A-010" in problems
    assert not any(
        "required E-A-010 coverage missing" in problem for problem in problems
    )


def test_full_gate_rejects_core_language_as_extended_lockout_coverage(tmp_path):
    truth, fixture = _write_fixture(tmp_path)
    truth.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "core-language-lockout",
                        "source_fixture": fixture.relative_to(tmp_path).as_posix(),
                        "source_sha256": checker._format_sha256(
                            checker._sha256(fixture)
                        ),
                        "expected_document_language": "hi",
                        "document_class": "extended-tier language lockout case",
                        "expected_page_languages": [
                            {
                                "page": 1,
                                "primary_language": "hi",
                                "languages_present": ["hi"],
                            }
                        ],
                        "extended_tier_lockout": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    problems = checker._full_gate_problems(tmp_path, truth)

    assert (
        "required E-A-010 coverage missing: extended-tier language lockout case"
        in problems
    )


def test_full_gate_passes_with_hashes_and_required_coverage(tmp_path, monkeypatch):
    truth, _fixture = _write_fixture(tmp_path)
    source_fixture_paths = []

    def document(doc_id, language, document_class, pages, **extra):
        fixture = tmp_path / "fixtures" / f"{doc_id}.txt"
        fixture.write_text(f"source for {doc_id}\n", encoding="utf-8")
        source_fixture_paths.append(fixture.relative_to(tmp_path).as_posix())
        return {
            "doc_id": doc_id,
            "source_fixture": fixture.relative_to(tmp_path).as_posix(),
            "source_sha256": checker._format_sha256(checker._sha256(fixture)),
            "expected_document_language": language,
            "document_class": document_class,
            "expected_page_languages": pages,
            **extra,
        }

    one_page_en = [{"page": 1, "primary_language": "en", "languages_present": ["en"]}]
    documents = [
        document(f"en-{i}", "en", "monolingual Latin document", one_page_en)
        for i in range(43)
    ]
    documents.extend(
        [
            document(
                "fr-1",
                "fr",
                "monolingual Latin document",
                [{"page": 1, "primary_language": "fr", "languages_present": ["fr"]}],
            ),
            document(
                "mixed-cjk",
                "mixed-cjk",
                "mixed CJK scan",
                [
                    {
                        "page": 1,
                        "primary_language": "mixed-cjk",
                        "languages_present": ["zh", "ja"],
                    }
                ],
            ),
            document(
                "rtl-ltr",
                "ar",
                "mixed RTL and LTR document",
                [
                    {
                        "page": 1,
                        "primary_language": "ar",
                        "languages_present": ["ar", "en"],
                    }
                ],
            ),
            document(
                "short-und",
                "und",
                "short noisy low-confidence document",
                [{"page": 1, "primary_language": "und", "languages_present": ["und"]}],
            ),
            document(
                "extended-lockout",
                "th",
                "extended-tier language lockout case",
                [{"page": 1, "primary_language": "th", "languages_present": ["th"]}],
                extended_tier_lockout=True,
            ),
            document(
                "page-transition",
                "en",
                "bilingual page transition",
                [
                    {"page": 1, "primary_language": "en", "languages_present": ["en"]},
                    {"page": 2, "primary_language": "fr", "languages_present": ["fr"]},
                ],
            ),
            document(
                "degraded",
                "en",
                "degraded low-resolution scan",
                one_page_en,
            ),
        ]
    )
    truth.write_text(json.dumps({"documents": documents}), encoding="utf-8")

    manifest_path = tmp_path / "corpus" / "language_truth.sha256"
    manifest = checker.build_manifest(
        tmp_path,
        truth,
        source_fixture_paths,
    )
    checker.write_manifest(manifest_path, manifest)

    args = [
        "--root",
        str(tmp_path),
        "--truth",
        str(truth),
        "--manifest",
        str(manifest_path),
        "--require-full-gate",
    ]
    for source_fixture_path in source_fixture_paths:
        args.extend(["--include", source_fixture_path])

    assert (
        checker.main(args) == 0
    )

    # An owner-approved update of a valid corpus is allowed only when the
    # explicit full-gate switch is present. The update replaces stale bytes
    # with the manifest computed from the same gate-validated inputs.
    manifest_path.write_text('{"stale": true}\n', encoding="utf-8")
    monkeypatch.setenv("CORPUS_GOLDEN_SET_UPDATE_APPROVED", "true")
    assert checker.main([*args, "--update"]) == 0
    assert checker.load_manifest(manifest_path) == checker.build_manifest(
        tmp_path,
        truth,
        source_fixture_paths,
    )
