#!/usr/bin/env python3
"""Verify the frozen Plan A language corpus manifest.

The Plan A E-A-010 gate freezes corpus/language_truth.json and its source
fixtures behind an explicit approval step. This checker fails when any tracked
file hash differs from corpus/language_truth.sha256 unless the operator runs an
approved manifest regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

# Running this file as ``python scripts/check_language_corpus_manifest.py``
# places ``scripts/`` rather than the repository root on sys.path. Bootstrap the
# root before importing project modules so the documented CLI works in a clean
# environment without requiring PYTHONPATH or an installed package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ocr_local.config.language_config import LANGUAGE_REGISTRY

_DEFAULT_TRUTH = _PROJECT_ROOT / "corpus" / "language_truth.json"
_DEFAULT_MANIFEST = _PROJECT_ROOT / "corpus" / "language_truth.sha256"
_DEFAULT_INCLUDES = (
    "corpus/language_fixtures/*.txt",
    "tests/fixtures/corpus/ground-truth/*.gt.txt",
    "tests/fixtures/corpus/output/*.txt",
)
_APPROVAL_ENV = "CORPUS_GOLDEN_SET_UPDATE_APPROVED"
_FULL_GATE_MIN_DOCUMENTS = 50
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMATTED_SHA256_RE = re.compile(r"^(?:[0-9a-f]{8}-){7}[0-9a-f]{8}$", re.IGNORECASE)


def _language_alias_map() -> dict[str, str | None]:
    """Map every registry spelling to one canonical Paddle language code.

    Ambiguous aliases deliberately map to ``None`` so coverage can never be
    granted by an alias whose meaning depends on registry iteration order.
    """
    candidates: dict[str, set[str]] = {}
    for paddle_code, entry in LANGUAGE_REGISTRY.items():
        aliases = (
            paddle_code,
            entry.paddle_code,
            *entry.fasttext_codes,
            entry.bcp47,
            entry.tesseract_code,
            entry.easyocr_code,
        )
        for alias in aliases:
            if alias:
                candidates.setdefault(alias.strip().lower(), set()).add(
                    paddle_code.strip().lower()
                )
    return {
        alias: next(iter(codes)) if len(codes) == 1 else None
        for alias, codes in candidates.items()
    }


_LANGUAGE_ALIAS_TO_CANONICAL = _language_alias_map()


def _canonical_language(value: str) -> str | None:
    """Return a registry language code for a supported alias, if unambiguous."""
    return _LANGUAGE_ALIAS_TO_CANONICAL.get(value.strip().lower())


_LATIN_LANGUAGES = frozenset(
    code.lower()
    for code, entry in LANGUAGE_REGISTRY.items()
    if entry.script == "latin"
)
_RTL_LANGUAGES = frozenset(
    code.lower() for code, entry in LANGUAGE_REGISTRY.items() if entry.rtl
)
_LTR_LANGUAGES = frozenset(
    code.lower() for code, entry in LANGUAGE_REGISTRY.items() if not entry.rtl
)
_CJK_LANGUAGES = frozenset(
    code.lower()
    for code, entry in LANGUAGE_REGISTRY.items()
    if entry.script == "cjk"
)
_SPECIAL_LANGUAGE_CODES = frozenset({"und", "mixed-cjk"})
_NEGATION_WORDS = frozenset({"no", "non", "not", "never", "without"})
_CLAUSE_BREAK_WORDS = frozenset({"and", "but", "or", "yet"})


def _normalize_sha256(value: object) -> str:
    return str(value).replace("-", "").lower()


def _is_sha256_declaration(value: object) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    return bool(
        _SHA256_RE.fullmatch(candidate.lower())
        or _FORMATTED_SHA256_RE.fullmatch(candidate)
    )


def _has_unnegated_phrase(words: list[str], phrase: str) -> bool:
    """Return whether a class phrase occurs without a nearby negation."""
    target = phrase.split()
    if not target:
        return False
    for start in range(len(words) - len(target) + 1):
        if words[start : start + len(target)] != target:
            continue
        context = words[:start]
        last_clause_break = max(
            (
                index
                for index, word in enumerate(context)
                if word in _CLAUSE_BREAK_WORDS
            ),
            default=-1,
        )
        context = context[last_clause_break + 1 :]
        if not _NEGATION_WORDS.intersection(context):
            return True
    return False


def _format_sha256(value: str) -> str:
    normalized = _normalize_sha256(value)
    if not _SHA256_RE.fullmatch(normalized):
        return value
    return "-".join(normalized[index : index + 8] for index in range(0, 64, 8))


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_includes(root: Path, truth: Path, includes: Iterable[str]) -> list[Path]:
    paths = {truth.resolve()}
    for pattern in includes:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"include pattern matched no files: {pattern}")
        paths.update(path.resolve() for path in matches if path.is_file())
    return sorted(paths, key=lambda p: _rel(p, root))


def build_manifest(
    root: Path, truth: Path, includes: Iterable[str]
) -> dict[str, object]:
    if not truth.is_file():
        raise FileNotFoundError(f"language truth file not found: {truth}")
    files = _expand_includes(root, truth, includes)
    entries = [
        ManifestEntry(
            path=_rel(path, root), sha256=_format_sha256(_sha256(path))
        ).__dict__
        for path in files
    ]
    composite = hashlib.sha256()
    for entry in entries:
        composite.update(entry["path"].encode("utf-8"))
        composite.update(b"\0")
        composite.update(_normalize_sha256(entry["sha256"]).encode("ascii"))
        composite.update(b"\n")
    return {
        "schema_version": "1.0",
        "gate": "Plan A E-A-010 frozen language corpus",
        "approval_env": _APPROVAL_ENV,
        "truth_path": _rel(truth, root),
        "composite_sha256": _format_sha256(composite.hexdigest()),
        "entries": entries,
    }


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def count_truth_documents(truth: Path) -> int:
    with truth.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise ValueError("language truth file must contain a documents list")
    return len(documents)


def _full_gate_problems(root: Path, truth: Path) -> list[str]:
    """Validate corpus size, fixture hashes, and the required coverage classes."""
    try:
        data = json.loads(truth.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read language truth data: {exc}"]

    if not isinstance(data, dict):
        return ["language truth root must be an object"]

    documents = data.get("documents")
    if not isinstance(documents, list):
        return ["language truth file must contain a documents list"]

    problems: list[str] = []
    if len(documents) < _FULL_GATE_MIN_DOCUMENTS:
        problems.append(
            f"{len(documents)} documents present; "
            f"{_FULL_GATE_MIN_DOCUMENTS} required for E-A-010"
        )

    document_ids = [
        row["doc_id"].strip()
        for row in documents
        if isinstance(row, dict)
        and isinstance(row.get("doc_id"), str)
        and row["doc_id"].strip()
    ]
    duplicate_ids = sorted(
        doc_id
        for doc_id in set(document_ids)
        if doc_id and document_ids.count(doc_id) > 1
    )
    if duplicate_ids:
        problems.append("duplicate document IDs: " + ", ".join(duplicate_ids))

    latin_monolingual = False
    mixed_cjk_page = False
    rtl_ltr_document = False
    short_noisy_und = False
    extended_lockout = False
    bilingual_page_transition = False
    degraded_scan = False

    root_resolved = root.resolve()
    source_fixture_owners: dict[Path, str] = {}
    for index, row in enumerate(documents):
        if not isinstance(row, dict):
            problems.append(f"document row {index + 1} is not an object")
            continue

        raw_doc_id = row.get("doc_id")
        if not isinstance(raw_doc_id, str) or not raw_doc_id.strip():
            problems.append(
                f"document row {index + 1}: doc_id must be a non-empty string"
            )
            doc_id = f"row-{index + 1}"
        else:
            doc_id = raw_doc_id.strip()
        fixture_value = row.get("source_fixture")
        fixture_name = fixture_value.strip() if isinstance(fixture_value, str) else ""
        if not fixture_name:
            problems.append(
                f"{doc_id}: source_fixture must be a non-empty relative path"
            )
        else:
            relative_fixture = PurePosixPath(fixture_name)
            if (
                relative_fixture.is_absolute()
                or ".." in relative_fixture.parts
                or "." in relative_fixture.parts
                or "\\" in fixture_name
                or str(relative_fixture) != fixture_name
            ):
                problems.append(
                    f"{doc_id}: source_fixture must be canonical repository-relative POSIX path"
                )
                relative_fixture = None
        hash_value = row.get("source_sha256")
        if not _is_sha256_declaration(hash_value):
            problems.append(f"{doc_id}: source_sha256 is missing or invalid")
            declared_hash = ""
        else:
            declared_hash = _normalize_sha256(hash_value)
        if fixture_name and relative_fixture is not None:
            try:
                fixture = (root_resolved / fixture_name).resolve()
                fixture.relative_to(root_resolved)
            except (OSError, RuntimeError, ValueError):
                problems.append(f"{doc_id}: source_fixture escapes repository root")
            else:
                if not fixture.is_file():
                    problems.append(
                        f"{doc_id}: source fixture is missing: {fixture_name}"
                    )
                else:
                    previous_doc_id = source_fixture_owners.get(fixture)
                    if previous_doc_id is not None:
                        problems.append(
                            f"{doc_id}: source_fixture duplicates document "
                            f"{previous_doc_id}: {fixture_name}"
                        )
                    else:
                        source_fixture_owners[fixture] = doc_id
                    if _sha256(fixture) != declared_hash:
                        problems.append(
                            f"{doc_id}: source_sha256 does not match fixture"
                        )

        expected_value = row.get("expected_document_language")
        if not isinstance(expected_value, str) or not expected_value.strip():
            problems.append(
                f"{doc_id}: expected_document_language must be a non-empty string"
            )
            expected = ""
        else:
            expected_raw = expected_value.strip().lower()
            expected = (
                expected_raw
                if expected_raw in _SPECIAL_LANGUAGE_CODES
                else _canonical_language(expected_raw) or ""
            )
            if not expected:
                problems.append(
                    f"{doc_id}: expected_document_language is not a supported language code: "
                    f"{expected_raw!r}"
                )
        class_value = row.get("document_class")
        if not isinstance(class_value, str) or not class_value.strip():
            problems.append(f"{doc_id}: document_class must be a non-empty string")
            document_class = ""
        else:
            document_class = class_value.strip().lower()
        class_phrase = re.sub(r"[-_]+", " ", document_class)
        class_word_list = re.findall(r"[a-z0-9]+", class_phrase)
        pages = row.get("expected_page_languages")
        if not isinstance(pages, list) or not pages:
            problems.append(
                f"{doc_id}: expected_page_languages must be a non-empty list"
            )
            pages = []
        page_primary: list[str] = []
        page_language_sets: list[set[str]] = []
        page_numbers: list[int] = []
        for page_index, page in enumerate(pages):
            page_label = f"{doc_id}: page row {page_index + 1}"
            if not isinstance(page, dict):
                problems.append(f"{page_label} must be an object")
                continue

            page_number = page.get("page")
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                problems.append(f"{page_label}: page must be a positive integer")
            else:
                page_numbers.append(page_number)

            primary_value = page.get("primary_language")
            if not isinstance(primary_value, str) or not primary_value.strip():
                problems.append(
                    f"{page_label}: primary_language must be a non-empty string"
                )
                primary = ""
            else:
                primary_raw = primary_value.strip().lower()
                primary = (
                    primary_raw
                    if primary_raw in _SPECIAL_LANGUAGE_CODES
                    else _canonical_language(primary_raw) or ""
                )
                if not primary:
                    problems.append(
                        f"{page_label}: primary_language is not a supported language code: "
                        f"{primary_raw!r}"
                    )
            page_primary.append(primary)

            languages_value = page.get("languages_present")
            if not isinstance(languages_value, list) or not languages_value:
                problems.append(
                    f"{page_label}: languages_present must be a non-empty list"
                )
                languages_value = []
            languages: set[str] = set()
            for language_index, language in enumerate(languages_value):
                if not isinstance(language, str) or not language.strip():
                    problems.append(
                        f"{page_label}: languages_present[{language_index}] "
                        "must be a non-empty string"
                    )
                    continue
                language_raw = language.strip().lower()
                canonical = (
                    "und"
                    if language_raw == "und"
                    else _canonical_language(language_raw)
                )
                if canonical is None:
                    problems.append(
                        f"{page_label}: languages_present[{language_index}] "
                        f"is not a supported language code: {language_raw!r}"
                    )
                    continue
                languages.add(canonical)
            if primary == "mixed-cjk":
                if len(languages & _CJK_LANGUAGES) < 2:
                    problems.append(
                        f"{page_label}: mixed-cjk primary requires at least two "
                        "supported CJK languages in languages_present"
                    )
            elif primary and primary not in languages:
                problems.append(
                    f"{page_label}: primary_language must appear in languages_present"
                )
            page_language_sets.append(languages)
        if len(page_numbers) != len(set(page_numbers)):
            problems.append(f"{doc_id}: page numbers must be unique")
        if page_numbers and page_numbers != sorted(page_numbers):
            problems.append(f"{doc_id}: page numbers must be in ascending order")
        document_languages: set[str] = set()
        for languages in page_language_sets:
            document_languages.update(languages)
        document_languages.discard("")

        if expected and page_primary and expected not in page_primary:
            problems.append(
                f"{doc_id}: expected_document_language must match at least one "
                "page primary_language"
            )

        if (
            expected in _LATIN_LANGUAGES
            and page_primary
            and all(language == expected for language in page_primary)
            and all(
                expected in languages and len(languages - {""}) == 1
                for languages in page_language_sets
            )
        ):
            latin_monolingual = True
        if any(
            len(languages & _CJK_LANGUAGES) >= 2 for languages in page_language_sets
        ):
            mixed_cjk_page = True
        if document_languages & _RTL_LANGUAGES and document_languages & _LTR_LANGUAGES:
            rtl_ltr_document = True
        if (
            expected == "und"
            and any(
                _has_unnegated_phrase(class_word_list, word)
                for word in ("short", "noisy")
            )
            and _has_unnegated_phrase(class_word_list, "low confidence")
        ):
            short_noisy_und = True
        has_extended_tier_language = any(
            LANGUAGE_REGISTRY.get(language) is not None
            and LANGUAGE_REGISTRY[language].tier == "extended"
            for languages in page_language_sets
            for language in languages
        )
        if has_extended_tier_language and (
            row.get("extended_tier_lockout") is True
            or _has_unnegated_phrase(
                class_word_list, "extended tier language lockout"
            )
        ):
            extended_lockout = True
        supported_page_primaries = {
            language for language in page_primary if language in LANGUAGE_REGISTRY
        }
        if len(supported_page_primaries) >= 2:
            bilingual_page_transition = True
        if any(
            _has_unnegated_phrase(class_word_list, phrase)
            for phrase in ("degraded", "lowres", "low resolution")
        ):
            degraded_scan = True

    unique_source_count = len(source_fixture_owners)
    if unique_source_count < _FULL_GATE_MIN_DOCUMENTS:
        problems.append(
            f"{unique_source_count} unique source fixtures present; "
            f"{_FULL_GATE_MIN_DOCUMENTS} required for E-A-010"
        )

    required_coverage = (
        ("monolingual Latin-script document", latin_monolingual),
        ("mixed CJK page", mixed_cjk_page),
        ("RTL + LTR mixed document", rtl_ltr_document),
        ("short/noisy low-confidence und document", short_noisy_und),
        ("extended-tier language lockout case", extended_lockout),
        ("bilingual/multilingual page transition", bilingual_page_transition),
        ("degraded/low-resolution scan", degraded_scan),
    )
    problems.extend(
        f"required E-A-010 coverage missing: {name}"
        for name, present in required_coverage
        if not present
    )
    return problems


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


def compare_manifest(
    expected: dict[str, object], actual: dict[str, object]
) -> list[str]:
    problems: list[str] = []
    if expected.get("truth_path") != actual.get("truth_path"):
        problems.append(
            f"truth_path mismatch: expected {expected.get('truth_path')!r}, "
            f"manifest has {actual.get('truth_path')!r}"
        )
    if _normalize_sha256(expected.get("composite_sha256")) != _normalize_sha256(
        actual.get("composite_sha256")
    ):
        problems.append("composite_sha256 mismatch")

    expected_entries = {
        entry["path"]: _normalize_sha256(entry["sha256"])
        for entry in expected["entries"]
    }
    actual_entries = {
        entry["path"]: _normalize_sha256(entry["sha256"])
        for entry in actual.get("entries", [])
    }
    for path in sorted(expected_entries.keys() - actual_entries.keys()):
        problems.append(f"manifest missing tracked file: {path}")
    for path in sorted(actual_entries.keys() - expected_entries.keys()):
        problems.append(f"manifest tracks removed or unexpected file: {path}")
    for path in sorted(expected_entries.keys() & actual_entries.keys()):
        if expected_entries[path] != actual_entries[path]:
            problems.append(f"sha256 mismatch for {path}")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--truth", type=Path, default=_DEFAULT_TRUTH)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="glob relative to --root for source fixtures to include in the manifest",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=f"regenerate the manifest; also requires {_APPROVAL_ENV}=true",
    )
    parser.add_argument(
        "--require-full-gate",
        action="store_true",
        help=(
            "require E-A-010 document count, per-fixture hashes, and coverage "
            f"({_FULL_GATE_MIN_DOCUMENTS} documents minimum)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    truth = args.truth if args.truth.is_absolute() else root / args.truth
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    )

    approved = os.environ.get(_APPROVAL_ENV, "").lower() == "true"
    if args.update and not approved:
        print(
            f"refusing to update {manifest_path}: set {_APPROVAL_ENV}=true "
            "for an explicit approved regeneration",
            file=sys.stderr,
        )
        return 1
    if args.update and not args.require_full_gate:
        print(
            f"refusing to update {manifest_path}: --update requires "
            "--require-full-gate",
            file=sys.stderr,
        )
        return 1

    if args.require_full_gate:
        full_gate_problems = _full_gate_problems(root, truth.resolve())
        if full_gate_problems:
            print("language corpus full-gate check failed:", file=sys.stderr)
            for problem in full_gate_problems:
                print(f"- {problem}", file=sys.stderr)
            return 1

    try:
        includes = args.include if args.include is not None else list(_DEFAULT_INCLUDES)
        current = build_manifest(root, truth.resolve(), includes)
    except FileNotFoundError as exc:
        print(f"language corpus manifest check failed: {exc}", file=sys.stderr)
        return 1

    if args.update:
        write_manifest(manifest_path, current)
        print(f"updated language corpus manifest: {manifest_path}")
        return 0

    if not manifest_path.is_file():
        print(
            f"language corpus manifest missing: {manifest_path}; regenerate with "
            f"--update and {_APPROVAL_ENV}=true",
            file=sys.stderr,
        )
        return 1

    recorded = load_manifest(manifest_path)
    problems = compare_manifest(current, recorded)
    if problems:
        print("language corpus manifest check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        print(
            f"regeneration requires --update and {_APPROVAL_ENV}=true",
            file=sys.stderr,
        )
        return 1

    print(f"language corpus manifest verified: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
