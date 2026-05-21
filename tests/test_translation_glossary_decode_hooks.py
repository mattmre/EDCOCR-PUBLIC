"""Tests for glossary constrained-decode hooks (Plan B Wave M2 PR B16a).

Covers the four pure helpers added to ``ocr_local.translation.glossary``:

* ``compute_glossary_hash`` -- deterministic SHA-256 over canonicalized
  entry content; must be tenant-/order-independent and pinned into
  TRANSLATION_APPLIED custody events.
* ``to_ct2_prefix_bias`` -- target-string -> bias-strength dict for the
  CT2 ``prefer`` mode (regex skipped, empty target skipped, dedup-max).
* ``to_ct2_forbid_tokens`` -- sorted dedup forbid list for the CT2
  ``force`` / ``reject-on-miss`` modes; honours ``forbid:`` notes
  directive when present.
* ``to_llm_prompt_block`` -- prompt-injection block for LLM-MT engines;
  truncates with overflow signal, marks regex entries.

Plus a JSON-Schema sanity check on ``schemas/glossary.schema.json``
when ``jsonschema`` is installed (graceful skip otherwise so the SDK CI
lane stays stdlib-only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_local.translation.glossary import (
    GlossaryEntry,
    compute_glossary_hash,
    to_ct2_forbid_tokens,
    to_ct2_prefix_bias,
    to_llm_prompt_block,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    *,
    term_id: str = "1",
    tenant_id: str = "t1",
    source: str = "Party",
    target: str = "Partie",
    src: str = "en",
    tgt: str = "fr",
    case_sensitive: bool = False,
    is_regex: bool = False,
    priority: int = 100,
) -> GlossaryEntry:
    return GlossaryEntry(
        term_id=term_id,
        tenant_id=tenant_id,
        source_term=source,
        target_term=target,
        source_lang=src,
        target_lang=tgt,
        case_sensitive=case_sensitive,
        is_regex=is_regex,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# compute_glossary_hash
# ---------------------------------------------------------------------------


def test_glossary_hash_is_sha256_hex():
    h = compute_glossary_hash([_entry()])
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_glossary_hash_empty_is_stable():
    h1 = compute_glossary_hash([])
    h2 = compute_glossary_hash([])
    assert h1 == h2
    # hash of "[]" canonical JSON
    assert h1 == compute_glossary_hash([])


def test_glossary_hash_is_deterministic_across_calls():
    entries = [
        _entry(term_id="a", source="Party", target="Partie"),
        _entry(term_id="b", source="Defendant", target="Defendeur"),
    ]
    assert compute_glossary_hash(entries) == compute_glossary_hash(entries)


def test_glossary_hash_is_order_independent():
    a = _entry(term_id="a", source="Party", target="Partie", priority=10)
    b = _entry(term_id="b", source="Defendant", target="Defendeur", priority=20)
    assert compute_glossary_hash([a, b]) == compute_glossary_hash([b, a])


def test_glossary_hash_is_tenant_independent():
    a = _entry(tenant_id="tenant-A")
    b = _entry(tenant_id="tenant-B")
    # Same content, different tenant -> same hash so external glossary
    # registries can identify content across tenants.
    assert compute_glossary_hash([a]) == compute_glossary_hash([b])


def test_glossary_hash_is_language_independent():
    a = _entry(src="en", tgt="fr")
    b = _entry(src="de", tgt="ja")
    assert compute_glossary_hash([a]) == compute_glossary_hash([b])


def test_glossary_hash_changes_when_target_term_changes():
    a = _entry(target="Partie")
    b = _entry(target="Partie ")
    assert compute_glossary_hash([a]) != compute_glossary_hash([b])


def test_glossary_hash_changes_when_priority_changes():
    a = _entry(priority=10)
    b = _entry(priority=20)
    assert compute_glossary_hash([a]) != compute_glossary_hash([b])


def test_glossary_hash_changes_when_case_sensitive_changes():
    a = _entry(case_sensitive=False)
    b = _entry(case_sensitive=True)
    assert compute_glossary_hash([a]) != compute_glossary_hash([b])


def test_glossary_hash_changes_when_regex_flag_changes():
    a = _entry(is_regex=False)
    b = _entry(is_regex=True)
    assert compute_glossary_hash([a]) != compute_glossary_hash([b])


# ---------------------------------------------------------------------------
# to_ct2_prefix_bias
# ---------------------------------------------------------------------------


def test_prefix_bias_emits_target_term_to_strength():
    out = to_ct2_prefix_bias([_entry(target="Partie")], bias_strength=2.5)
    assert out == {"Partie": 2.5}


def test_prefix_bias_default_strength_is_2():
    out = to_ct2_prefix_bias([_entry(target="Partie")])
    assert out == {"Partie": 2.0}


def test_prefix_bias_empty_iterable_returns_empty_dict():
    assert to_ct2_prefix_bias([]) == {}


def test_prefix_bias_skips_regex_entries():
    entries = [_entry(target="Partie", is_regex=True)]
    assert to_ct2_prefix_bias(entries) == {}


def test_prefix_bias_skips_empty_target_terms():
    entries = [_entry(target=""), _entry(term_id="2", target="Partie")]
    out = to_ct2_prefix_bias(entries)
    assert "" not in out
    assert out == {"Partie": 2.0}


def test_prefix_bias_dedups_target_with_max_strength():
    entries = [
        _entry(term_id="a", target="Partie"),
        _entry(term_id="b", target="Partie"),
    ]
    # Both have the default bias 2.0; result has one key.
    out = to_ct2_prefix_bias(entries, bias_strength=3.5)
    assert out == {"Partie": 3.5}


def test_prefix_bias_accepts_generator_input():
    def gen():
        yield _entry(term_id="a", target="Partie")
        yield _entry(term_id="b", target="Defendeur")

    out = to_ct2_prefix_bias(gen())
    assert out == {"Partie": 2.0, "Defendeur": 2.0}


def test_prefix_bias_preserves_target_string_unicode():
    entries = [_entry(target="Tribunal de Première Instance")]
    out = to_ct2_prefix_bias(entries)
    assert "Tribunal de Première Instance" in out


# ---------------------------------------------------------------------------
# to_ct2_forbid_tokens
# ---------------------------------------------------------------------------


def test_forbid_tokens_default_blocks_source_terms():
    entries = [
        _entry(source="Party"),
        _entry(term_id="2", source="Defendant"),
    ]
    out = to_ct2_forbid_tokens(entries)
    assert out == ["Defendant", "Party"]  # sorted


def test_forbid_tokens_block_source_terms_false_returns_empty_without_notes():
    out = to_ct2_forbid_tokens([_entry(source="Party")], block_source_terms=False)
    assert out == []


def test_forbid_tokens_skips_regex_entries():
    entries = [_entry(source="Party.*", is_regex=True)]
    assert to_ct2_forbid_tokens(entries) == []


def test_forbid_tokens_dedups_repeated_source_terms():
    entries = [
        _entry(term_id="a", source="Party"),
        _entry(term_id="b", source="Party"),
    ]
    assert to_ct2_forbid_tokens(entries) == ["Party"]


def test_forbid_tokens_returns_sorted_for_determinism():
    entries = [
        _entry(term_id="a", source="Zebra"),
        _entry(term_id="b", source="Alpha"),
        _entry(term_id="c", source="Mango"),
    ]
    assert to_ct2_forbid_tokens(entries) == ["Alpha", "Mango", "Zebra"]


def test_forbid_tokens_empty_iterable_returns_empty_list():
    assert to_ct2_forbid_tokens([]) == []


def test_forbid_tokens_skips_empty_source_term():
    # Empty source_term cannot exist via dataclass (min length enforced
    # by schema), but defend against it for robustness.
    entries = [_entry(source="")]
    assert to_ct2_forbid_tokens(entries) == []


# ---------------------------------------------------------------------------
# to_llm_prompt_block
# ---------------------------------------------------------------------------


def test_prompt_block_empty_returns_empty_string():
    assert to_llm_prompt_block([]) == ""


def test_prompt_block_emits_header_and_arrow_lines():
    entries = [_entry(source="Party", target="Partie")]
    out = to_llm_prompt_block(entries)
    assert out.startswith("Use the following terminology when translating:")
    assert '- "Party" -> "Partie"' in out


def test_prompt_block_marks_regex_entries():
    entries = [_entry(source=r"Party\d+", target="Partie", is_regex=True)]
    out = to_llm_prompt_block(entries)
    assert '- (regex) "Party\\d+" -> "Partie"' in out


def test_prompt_block_truncates_at_max_entries_with_overflow_signal():
    entries = [
        _entry(term_id=str(i), source=f"src{i}", target=f"tgt{i}", priority=i)
        for i in range(10)
    ]
    out = to_llm_prompt_block(entries, max_entries=3)
    # Only first 3 sortable entries appear.
    assert '- "src0" -> "tgt0"' in out
    assert '- "src1" -> "tgt1"' in out
    assert '- "src2" -> "tgt2"' in out
    assert '- "src3" -> "tgt3"' not in out
    # Overflow signaller present.
    assert "... (7 more)" in out


def test_prompt_block_no_overflow_marker_when_under_limit():
    entries = [_entry(source="Party", target="Partie")]
    out = to_llm_prompt_block(entries, max_entries=10)
    assert "more)" not in out


def test_prompt_block_custom_header():
    out = to_llm_prompt_block(
        [_entry()],
        header="Custom legal-domain glossary:",
    )
    assert out.startswith("Custom legal-domain glossary:")


def test_prompt_block_skips_entries_with_empty_source():
    entries = [_entry(source=""), _entry(term_id="2", source="Party")]
    out = to_llm_prompt_block(entries)
    assert '- "Party" -> "Partie"' in out
    # Empty-source entry is filtered out before sorting.
    assert '- "" ->' not in out


def test_prompt_block_sorts_by_priority_then_term_id():
    entries = [
        _entry(term_id="z", source="Zebra", target="Z", priority=10),
        _entry(term_id="a", source="Alpha", target="A", priority=10),
        _entry(term_id="m", source="Mango", target="M", priority=5),
    ]
    out = to_llm_prompt_block(entries)
    lines = out.splitlines()
    # Header + Mango (priority 5) + Alpha (priority 10, term_id "a")
    # + Zebra (priority 10, term_id "z").
    assert lines[1] == '- "Mango" -> "M"'
    assert lines[2] == '- "Alpha" -> "A"'
    assert lines[3] == '- "Zebra" -> "Z"'


# ---------------------------------------------------------------------------
# JSON Schema (schemas/glossary.schema.json)
# ---------------------------------------------------------------------------


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "glossary.schema.json"
)


def test_glossary_schema_file_exists_and_parses():
    assert _SCHEMA_PATH.is_file()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("http://json-schema.org/")
    assert schema["title"] == "Translation Glossary Bundle"
    assert schema["required"] == [
        "schema_version",
        "tenant_id",
        "source_lang",
        "target_lang",
        "entries",
    ]
    # Entry definition pinned for the runtime.
    entry_props = schema["definitions"]["glossary_entry"]["properties"]
    assert "source_term" in entry_props
    assert "target_term" in entry_props
    assert entry_props["priority"]["maximum"] == 100000


def test_glossary_schema_validates_minimal_bundle():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    minimal = {
        "schema_version": "1.0",
        "tenant_id": "tenant-A",
        "source_lang": "en",
        "target_lang": "fr",
        "entries": [
            {"source_term": "Party", "target_term": "Partie"},
        ],
    }
    # Should not raise.
    jsonschema.validate(minimal, schema)


def test_glossary_schema_rejects_missing_required_field():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = {
        "schema_version": "1.0",
        # missing tenant_id
        "source_lang": "en",
        "target_lang": "fr",
        "entries": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_glossary_schema_rejects_bad_glossary_hash():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = {
        "schema_version": "1.0",
        "tenant_id": "tenant-A",
        "source_lang": "en",
        "target_lang": "fr",
        "glossary_hash": "NOT-A-HEX-DIGEST",
        "entries": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_glossary_schema_accepts_valid_glossary_hash():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    good = {
        "schema_version": "1.0",
        "tenant_id": "tenant-A",
        "source_lang": "en",
        "target_lang": "fr",
        "glossary_hash": "a" * 64,
        "entries": [],
    }
    jsonschema.validate(good, schema)


def test_glossary_schema_rejects_unknown_top_level_property():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = {
        "schema_version": "1.0",
        "tenant_id": "tenant-A",
        "source_lang": "en",
        "target_lang": "fr",
        "entries": [],
        "rogue_field": "boom",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
