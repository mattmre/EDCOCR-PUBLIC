"""Tests for layoutlm_labels – BIO label management for LayoutLMv3."""

import json

import pytest

from layoutlm_labels import (
    BUILTIN_LABEL_SETS,
    DEFAULT_TYPE_MAP,
    build_label_set,
    expand_to_bio,
    get_active_label_set,
    load_label_set,
)

# ---------------------------------------------------------------------------
# The legacy 19-label list as defined in semantic_extraction.py.
# This is the canonical reference for backward-compatibility tests.
# ---------------------------------------------------------------------------

LEGACY_19_LABELS = [
    "O",
    "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
    "B-DATE", "I-DATE",
    "B-AMOUNT", "I-AMOUNT",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-ADDRESS", "I-ADDRESS",
    "B-REFERENCE_NUMBER", "I-REFERENCE_NUMBER",
    "B-PHONE_NUMBER", "I-PHONE_NUMBER",
    "B-EMAIL", "I-EMAIL",
]


# ── expand_to_bio ─────────────────────────────────────────────────────────


class TestExpandToBio:
    """Tests for the expand_to_bio helper."""

    def test_basic_expansion(self):
        """A single entity type yields [O, B-TYPE, I-TYPE]."""
        result = expand_to_bio(["DATE"])
        assert result == ["O", "B-DATE", "I-DATE"]

    def test_empty_list(self):
        """An empty entity list yields only the O tag."""
        assert expand_to_bio([]) == ["O"]

    def test_ordering_preserved(self):
        """Entity ordering is preserved in BIO output."""
        result = expand_to_bio(["AMOUNT", "DATE"])
        assert result == ["O", "B-AMOUNT", "I-AMOUNT", "B-DATE", "I-DATE"]

    def test_no_duplicates_after_expansion(self):
        """All BIO labels are unique when inputs are unique."""
        types = ["A", "B", "C"]
        result = expand_to_bio(types)
        assert len(result) == len(set(result)), "Duplicate labels detected"

    def test_label_count_formula(self):
        """Number of labels = 2 * len(entity_types) + 1."""
        for n in (0, 1, 5, 20):
            types = [f"T{i}" for i in range(n)]
            assert len(expand_to_bio(types)) == 2 * n + 1


# ── Built-in label sets ───────────────────────────────────────────────────


class TestBuiltinLabelSets:
    """Tests for BUILTIN_LABEL_SETS and their build_label_set output."""

    def test_all_four_sets_exist(self):
        """All four named sets are defined."""
        expected = {"default", "forensic", "receipt", "form"}
        assert expected == set(BUILTIN_LABEL_SETS.keys())

    def test_default_set_exact_19_labels(self):
        """The default set produces exactly the legacy 19 BIO labels."""
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        assert list(ls.bio_labels) == LEGACY_19_LABELS
        assert ls.num_labels == 19

    def test_forensic_contains_case_number(self):
        """The forensic set includes CASE_NUMBER entity type."""
        assert "CASE_NUMBER" in BUILTIN_LABEL_SETS["forensic"]
        ls = build_label_set("forensic", BUILTIN_LABEL_SETS["forensic"])
        assert "B-CASE_NUMBER" in ls.bio_labels

    def test_receipt_contains_total(self):
        """The receipt set includes TOTAL entity type."""
        assert "TOTAL" in BUILTIN_LABEL_SETS["receipt"]

    def test_form_contains_checkbox(self):
        """The form set includes CHECKBOX entity type."""
        assert "CHECKBOX" in BUILTIN_LABEL_SETS["form"]

    def test_num_labels_formula_all_sets(self):
        """num_labels = 2 * len(entity_types) + 1 for every built-in set."""
        for name, etypes in BUILTIN_LABEL_SETS.items():
            ls = build_label_set(name, etypes)
            assert ls.num_labels == 2 * len(etypes) + 1, (
                f"Mismatch for {name}"
            )


# ── label2id / id2label roundtrip ─────────────────────────────────────────


class TestLabelIdRoundtrip:
    """Tests for bijection between label2id and id2label."""

    def test_bijection_default(self):
        """label2id and id2label are inverses for the default set."""
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        for label, idx in ls.label2id.items():
            assert ls.id2label[idx] == label
        for idx, label in ls.id2label.items():
            assert ls.label2id[label] == idx

    def test_lengths_match(self):
        """label2id and id2label have the same length as bio_labels."""
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        assert len(ls.label2id) == ls.num_labels
        assert len(ls.id2label) == ls.num_labels


# ── LabelSet immutability ─────────────────────────────────────────────────


class TestLabelSetImmutability:
    """LabelSet instances should be frozen (dataclass frozen=True)."""

    def test_cannot_reassign_name(self):
        """Attempting to set an attribute raises FrozenInstanceError."""
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        with pytest.raises(AttributeError):
            ls.name = "hacked"

    def test_cannot_reassign_num_labels(self):
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        with pytest.raises(AttributeError):
            ls.num_labels = 999


# ── load_label_set ────────────────────────────────────────────────────────


class TestLoadLabelSet:
    """Tests for load_label_set (built-in + custom JSON)."""

    def test_load_builtin_by_name(self):
        """Loading a known built-in name returns a valid LabelSet."""
        ls = load_label_set("default")
        assert ls.name == "default"
        assert ls.num_labels == 19

    def test_unknown_name_raises(self):
        """An unrecognised name (non-.json) raises ValueError."""
        with pytest.raises(ValueError, match="Unknown label set"):
            load_label_set("nonexistent")

    def test_missing_json_file_raises(self, tmp_path):
        """A .json path that does not exist raises FileNotFoundError."""
        missing = str(tmp_path / "nope.json")
        with pytest.raises(FileNotFoundError):
            load_label_set(missing)

    def test_custom_json_loading(self, tmp_path):
        """A valid JSON file is loaded into a LabelSet correctly."""
        cfg = {
            "name": "custom_test",
            "entity_types": ["ALPHA", "BETA"],
            "type_map": {"ALPHA": "alpha_field"},
        }
        json_path = tmp_path / "custom.json"
        json_path.write_text(json.dumps(cfg), encoding="utf-8")

        ls = load_label_set(str(json_path))
        assert ls.name == "custom_test"
        assert list(ls.entity_types) == ["ALPHA", "BETA"]
        assert ls.num_labels == 5  # O + 2*2
        assert ls.type_map["ALPHA"] == "alpha_field"
        # BETA should fall back to DEFAULT_TYPE_MAP or lowercase
        assert ls.type_map["BETA"] == "beta"

    def test_custom_json_without_name(self, tmp_path):
        """If name is omitted in JSON the file stem is used."""
        cfg = {"entity_types": ["GAMMA"]}
        json_path = tmp_path / "my_labels.json"
        json_path.write_text(json.dumps(cfg), encoding="utf-8")

        ls = load_label_set(str(json_path))
        assert ls.name == "my_labels"


# ── get_active_label_set (env-var driven) ─────────────────────────────────


class TestGetActiveLabelSet:
    """Tests for get_active_label_set with env-var overrides."""

    def test_default_when_no_env(self, monkeypatch):
        """Without any env vars, the default set is returned."""
        monkeypatch.delenv("LAYOUTLM_LABEL_SET", raising=False)
        monkeypatch.delenv("LAYOUTLM_LABEL_CONFIG", raising=False)
        ls = get_active_label_set()
        assert ls.name == "default"

    def test_env_selects_forensic(self, monkeypatch):
        """LAYOUTLM_LABEL_SET=forensic activates the forensic set."""
        monkeypatch.setenv("LAYOUTLM_LABEL_SET", "forensic")
        monkeypatch.delenv("LAYOUTLM_LABEL_CONFIG", raising=False)
        ls = get_active_label_set()
        assert ls.name == "forensic"
        assert "CASE_NUMBER" in ls.entity_types

    def test_env_config_overrides_set(self, monkeypatch, tmp_path):
        """LAYOUTLM_LABEL_CONFIG takes precedence over LAYOUTLM_LABEL_SET."""
        cfg = {"name": "env_custom", "entity_types": ["X"]}
        json_path = tmp_path / "env_custom.json"
        json_path.write_text(json.dumps(cfg), encoding="utf-8")

        monkeypatch.setenv("LAYOUTLM_LABEL_SET", "forensic")
        monkeypatch.setenv("LAYOUTLM_LABEL_CONFIG", str(json_path))
        ls = get_active_label_set()
        assert ls.name == "env_custom"

    def test_env_unknown_set_raises(self, monkeypatch):
        """An invalid LAYOUTLM_LABEL_SET raises ValueError."""
        monkeypatch.setenv("LAYOUTLM_LABEL_SET", "does_not_exist")
        monkeypatch.delenv("LAYOUTLM_LABEL_CONFIG", raising=False)
        with pytest.raises(ValueError):
            get_active_label_set()


# ── type_map coverage ─────────────────────────────────────────────────────


class TestTypeMap:
    """Tests for DEFAULT_TYPE_MAP and per-set type_map resolution."""

    def test_default_type_map_covers_default_set(self):
        """Every entity in the default set has a DEFAULT_TYPE_MAP entry."""
        for etype in BUILTIN_LABEL_SETS["default"]:
            assert etype in DEFAULT_TYPE_MAP, f"{etype} missing from DEFAULT_TYPE_MAP"

    def test_build_label_set_uses_default_type_map(self):
        """build_label_set falls back to DEFAULT_TYPE_MAP entries."""
        ls = build_label_set("default", BUILTIN_LABEL_SETS["default"])
        assert ls.type_map["DATE"] == "date"
        assert ls.type_map["EMAIL"] == "email_address"

    def test_custom_type_map_overrides_default(self):
        """Caller-supplied type_map overrides DEFAULT_TYPE_MAP."""
        ls = build_label_set(
            "custom",
            ["DATE"],
            type_map={"DATE": "custom_date"},
        )
        assert ls.type_map["DATE"] == "custom_date"
