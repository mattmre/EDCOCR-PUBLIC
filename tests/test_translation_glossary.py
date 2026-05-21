"""Tests for ``ocr_local.translation.glossary`` (Plan B Wave M2).

Pure-Python tests for ``apply_glossary`` and its helpers -- no Django
required.  Tests for the Django-backed ``load_tenant_glossary`` are
guarded with ``pytest.importorskip("django")`` and use Django fixtures.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ocr_local.translation.glossary import (
    GlossaryEntry,
    GlossaryHit,
    apply_glossary,
    emit_glossary_applied_for_span,
    load_tenant_glossary,
)


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
# apply_glossary -- literal entries
# ---------------------------------------------------------------------------


def test_apply_glossary_no_entries_returns_input():
    text = "The Party agrees..."
    out, hits = apply_glossary(text, [])
    assert out == text
    assert hits == []


def test_apply_glossary_no_text_returns_empty():
    out, hits = apply_glossary("", [_entry()])
    assert out == ""
    assert hits == []


def test_apply_glossary_literal_case_insensitive_default():
    text = "The Party and the party agree."
    out, hits = apply_glossary(text, [_entry(source="party", target="Partie")])
    assert out == "The Partie and the Partie agree."
    assert len(hits) == 2


def test_apply_glossary_literal_case_sensitive():
    text = "The Party and the party agree."
    out, hits = apply_glossary(
        text,
        [_entry(source="Party", target="Partie", case_sensitive=True)],
    )
    assert out == "The Partie and the party agree."
    assert len(hits) == 1
    assert hits[0].position_start == 4
    assert hits[0].position_end == 9


def test_apply_glossary_no_match_returns_input_unchanged():
    text = "Nothing matches here."
    out, hits = apply_glossary(text, [_entry(source="zzz", target="qqq")])
    assert out == text
    assert hits == []


def test_apply_glossary_priority_ordering_applies_in_input_order():
    """Entries are applied sequentially in the order provided.

    ``load_tenant_glossary`` pre-sorts by priority so the runtime helper
    does not re-sort.  This test pins the contract.
    """
    text = "Party Agreement"
    entries = [
        _entry(term_id="1", source="Party", target="Partie", priority=10),
        _entry(term_id="2", source="Agreement", target="Accord", priority=20),
    ]
    out, hits = apply_glossary(text, entries)
    assert out == "Partie Accord"
    assert [h.term_id for h in hits] == ["1", "2"]


def test_apply_glossary_chained_replacement_within_span():
    """Each entry runs against the *current* text (chaining)."""
    text = "alpha"
    entries = [
        _entry(term_id="1", source="alpha", target="beta"),
        _entry(term_id="2", source="beta", target="gamma"),
    ]
    out, _hits = apply_glossary(text, entries)
    assert out == "gamma"


def test_apply_glossary_records_span_index():
    text = "Party"
    out, hits = apply_glossary(
        text, [_entry(source="Party", target="Partie")], span_index=7,
    )
    assert out == "Partie"
    assert hits[0].span_index == 7


def test_apply_glossary_hit_position_recorded_pre_mutation():
    """First hit's position uses the index in the original text."""
    text = "X Party Y"
    _out, hits = apply_glossary(text, [_entry(source="Party", target="Partie")])
    assert hits[0].position_start == 2
    assert hits[0].position_end == 7


def test_apply_glossary_empty_source_term_skipped():
    """Empty source term is a no-op (does not emit hits)."""
    out, hits = apply_glossary("anything", [_entry(source="", target="x")])
    assert out == "anything"
    assert hits == []


# ---------------------------------------------------------------------------
# apply_glossary -- regex entries
# ---------------------------------------------------------------------------


def test_apply_glossary_regex_basic():
    text = "Order #123 and Order #456"
    out, hits = apply_glossary(
        text,
        [_entry(source=r"#\d+", target="(redacted)", is_regex=True)],
    )
    assert out == "Order (redacted) and Order (redacted)"
    assert len(hits) == 2


def test_apply_glossary_regex_case_insensitive_default():
    out, hits = apply_glossary(
        "FOO Foo foo",
        [_entry(source=r"foo", target="bar", is_regex=True)],
    )
    assert out == "bar bar bar"
    assert len(hits) == 3


def test_apply_glossary_regex_case_sensitive():
    out, hits = apply_glossary(
        "FOO Foo foo",
        [_entry(source=r"foo", target="bar", is_regex=True, case_sensitive=True)],
    )
    assert out == "FOO Foo bar"
    assert len(hits) == 1


def test_apply_glossary_invalid_regex_skipped_no_raise():
    text = "abc"
    out, hits = apply_glossary(
        text,
        [_entry(source=r"[unclosed", target="x", is_regex=True)],
    )
    assert out == text
    assert hits == []


def test_apply_glossary_regex_input_too_large_skipped():
    """Inputs over the char budget are skipped (ReDoS guard)."""
    big_text = "a" * 200_001
    out, hits = apply_glossary(
        big_text,
        [_entry(source=r"a+", target="x", is_regex=True)],
    )
    assert out == big_text
    assert hits == []


def test_apply_glossary_regex_backref_supported():
    """Regex target_term supports backreferences via match.expand."""
    out, hits = apply_glossary(
        "USD 100 and USD 250",
        [_entry(source=r"USD (\d+)", target=r"\1 USD", is_regex=True)],
    )
    assert out == "100 USD and 250 USD"
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# load_tenant_glossary -- non-Django fallback
# ---------------------------------------------------------------------------


def test_load_tenant_glossary_none_tenant_returns_empty():
    assert load_tenant_glossary(None, "en", "fr") == []


def test_load_tenant_glossary_no_django_returns_empty(monkeypatch):
    """When Django/jobs.models is not importable, return [] gracefully."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("jobs.models") or name == "jobs.models":
            raise ImportError("simulated -- Django not configured")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    out = load_tenant_glossary("tenant-a", "en", "fr")
    assert out == []


# ---------------------------------------------------------------------------
# emit_glossary_applied_for_span
# ---------------------------------------------------------------------------


def test_emit_glossary_applied_for_span_no_hits_is_noop():
    chain = MagicMock()
    emit_glossary_applied_for_span(
        chain, glossary_id="g1", glossary_hash="h1", hits=[],
    )
    assert not chain.log_event.called


def test_emit_glossary_applied_for_span_emits_once():
    """A span with multiple hits emits exactly one event."""
    chain = MagicMock()
    hits = [
        GlossaryHit(
            term_id="1",
            source_term="Party",
            target_term="Partie",
            position_start=0,
            position_end=5,
            span_index=0,
        ),
        GlossaryHit(
            term_id="2",
            source_term="Agreement",
            target_term="Accord",
            position_start=6,
            position_end=15,
            span_index=0,
        ),
    ]
    emit_glossary_applied_for_span(
        chain,
        glossary_id="g1",
        glossary_hash="abc123",
        hits=hits,
        span_id="span_0",
    )
    assert chain.log_event.call_count == 1
    args, _kwargs = chain.log_event.call_args
    assert args[0] == "GLOSSARY_APPLIED"
    payload = args[1]
    assert payload["glossary_id"] == "g1"
    assert payload["glossary_hash"] == "abc123"
    assert payload["hit_count"] == 2
    assert payload["span_id"] == "span_0"
    assert "glossary_hits" in payload
    assert len(payload["glossary_hits"]) == 2
    # Each hit serialized as dict, not dataclass.
    assert payload["glossary_hits"][0]["term_id"] == "1"
    assert payload["glossary_hits"][0]["source_term"] == "Party"


def test_emit_glossary_applied_for_span_extra_kwargs_propagated():
    chain = MagicMock()
    hits = [
        GlossaryHit("1", "x", "y", 0, 1, 0),
    ]
    emit_glossary_applied_for_span(
        chain,
        glossary_id="g1",
        glossary_hash="abc123",
        hits=hits,
        tenant_id="t1",
    )
    args, _kwargs = chain.log_event.call_args
    assert args[1]["tenant_id"] == "t1"


# ---------------------------------------------------------------------------
# GlossaryEntry / GlossaryHit dataclass shape
# ---------------------------------------------------------------------------


def test_glossary_entry_is_frozen():
    entry = _entry()
    with pytest.raises(Exception):  # FrozenInstanceError on dataclass
        entry.priority = 5  # type: ignore[misc]


def test_glossary_hit_is_frozen():
    hit = GlossaryHit("1", "x", "y", 0, 1, 0)
    with pytest.raises(Exception):
        hit.position_start = 99  # type: ignore[misc]


def test_glossary_entry_default_priority_is_100():
    entry = GlossaryEntry(
        term_id="1",
        tenant_id="t1",
        source_term="x",
        target_term="y",
        source_lang="en",
        target_lang="fr",
    )
    assert entry.priority == 100
    assert entry.case_sensitive is False
    assert entry.is_regex is False
