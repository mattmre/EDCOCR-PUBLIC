"""Tests for schemas/translation.schema.json -- Plan B Wave M1 PR2."""

from __future__ import annotations

import json
import os

import pytest

_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "schemas", "translation.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    with open(os.path.normpath(_SCHEMA_PATH), encoding="utf-8") as f:
        return json.load(f)


def test_schema_file_exists():
    assert os.path.exists(os.path.normpath(_SCHEMA_PATH))


def test_schema_is_valid_json():
    with open(os.path.normpath(_SCHEMA_PATH), encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_schema_draft07(schema):
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"


def test_schema_has_required_fields(schema):
    required = set(schema["required"])
    assert {"certified", "source_language", "target_language", "pages"}.issubset(required)


def test_schema_certified_is_boolean(schema):
    assert schema["properties"]["certified"]["type"] == "boolean"


def test_schema_pages_is_array(schema):
    assert schema["properties"]["pages"]["type"] == "array"


def test_schema_custody_has_chain_head(schema):
    custody = schema["properties"]["custody"]
    assert "chain_head" in custody["properties"]


def test_schema_engine_has_license(schema):
    engine = schema["properties"]["engine"]
    assert "license" in engine["properties"]
