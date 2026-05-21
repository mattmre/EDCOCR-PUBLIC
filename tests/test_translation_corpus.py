"""Tests for the FLORES-200 translation test corpus loader."""

from __future__ import annotations

import importlib.util
import os

_FIXTURE_ROOT = os.path.join(
    os.path.dirname(__file__), "fixtures", "translation-corpus"
)


def _load_module():
    """Import the hyphenated package by file path."""

    spec = importlib.util.spec_from_file_location(
        "translation_corpus_loader",
        os.path.join(_FIXTURE_ROOT, "__init__.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flores_readme_exists():
    assert os.path.exists(os.path.join(_FIXTURE_ROOT, "README.md"))


def test_flores_en_fr_file_exists():
    path = os.path.join(_FIXTURE_ROOT, "flores200", "en_fr.tsv")
    assert os.path.exists(path)


def test_load_en_fr_returns_pairs():
    mod = _load_module()
    pairs = mod.load_flores_pairs("en", "fr")
    assert pairs
    assert isinstance(pairs, list)


def test_load_pairs_count():
    mod = _load_module()
    pairs = mod.load_flores_pairs("en", "fr")
    assert len(pairs) >= 10
    pairs_es = mod.load_flores_pairs("en", "es")
    assert len(pairs_es) >= 10


def test_load_pairs_are_tuples():
    mod = _load_module()
    pairs = mod.load_flores_pairs("en", "fr")
    for item in pairs:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], str)


def test_load_nonexistent_returns_empty():
    mod = _load_module()
    assert mod.load_flores_pairs("xx", "yy") == []
