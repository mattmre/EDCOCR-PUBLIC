"""Tests for env_utils.py centralized env schema and typed accessors."""

from __future__ import annotations

import logging
import os

import pytest

from env_utils import (
    ENV_SCHEMA,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    validate_env,
)

# ---------------------------------------------------------------------------
# get_env
# ---------------------------------------------------------------------------


def test_get_env_returns_value(monkeypatch):
    monkeypatch.setenv("_TEST_ENV_STR", "hello")
    assert get_env("_TEST_ENV_STR") == "hello"


def test_get_env_returns_default_when_missing():
    assert get_env("_TEST_NONEXISTENT_VAR_XYZ", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# get_env_int
# ---------------------------------------------------------------------------


def test_get_env_int_valid(monkeypatch):
    monkeypatch.setenv("_TEST_INT_VAR", "42")
    assert get_env_int("_TEST_INT_VAR", 0) == 42


def test_get_env_int_invalid_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("_TEST_INT_VAR", "not_a_number")
    with caplog.at_level(logging.WARNING):
        result = get_env_int("_TEST_INT_VAR", 99)
    assert result == 99
    assert "not a valid int" in caplog.text


def test_get_env_int_missing_returns_default():
    key = "_TEST_MISSING_INT_" + os.urandom(4).hex()
    assert get_env_int(key, 7) == 7


# ---------------------------------------------------------------------------
# get_env_float
# ---------------------------------------------------------------------------


def test_get_env_float_valid(monkeypatch):
    monkeypatch.setenv("_TEST_FLOAT_VAR", "3.14")
    assert get_env_float("_TEST_FLOAT_VAR", 0.0) == pytest.approx(3.14)


def test_get_env_float_invalid_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("_TEST_FLOAT_VAR", "abc")
    with caplog.at_level(logging.WARNING):
        result = get_env_float("_TEST_FLOAT_VAR", 1.5)
    assert result == pytest.approx(1.5)
    assert "not a valid float" in caplog.text


def test_get_env_float_missing_returns_default():
    key = "_TEST_MISSING_FLOAT_" + os.urandom(4).hex()
    assert get_env_float(key, 2.5) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# get_env_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["1", "true", "True", "TRUE", "yes", "Yes", "YES"],
)
def test_get_env_bool_truthy(monkeypatch, value):
    monkeypatch.setenv("_TEST_BOOL_VAR", value)
    assert get_env_bool("_TEST_BOOL_VAR") is True


@pytest.mark.parametrize(
    "value",
    ["0", "false", "False", "no", "No", "", "random"],
)
def test_get_env_bool_falsy(monkeypatch, value):
    monkeypatch.setenv("_TEST_BOOL_VAR", value)
    assert get_env_bool("_TEST_BOOL_VAR") is False


def test_get_env_bool_missing_returns_default():
    key = "_TEST_MISSING_BOOL_" + os.urandom(4).hex()
    assert get_env_bool(key, True) is True
    assert get_env_bool(key, False) is False


# ---------------------------------------------------------------------------
# validate_env
# ---------------------------------------------------------------------------


def test_validate_env_catches_bad_int(monkeypatch):
    monkeypatch.setenv("NUM_WORKERS", "not_int")
    warnings = validate_env()
    matching = [
        w for w in warnings if "NUM_WORKERS" in w and "cannot be parsed as int" in w
    ]
    assert len(matching) == 1


def test_validate_env_catches_bad_float(monkeypatch):
    monkeypatch.setenv("SSE_POLL_INTERVAL", "not_float")
    warnings = validate_env()
    matching = [
        w
        for w in warnings
        if "SSE_POLL_INTERVAL" in w and "cannot be parsed as float" in w
    ]
    assert len(matching) == 1


def test_validate_env_catches_bad_bool(monkeypatch):
    monkeypatch.setenv("ENABLE_DOCINTEL", "maybe")
    warnings = validate_env()
    matching = [
        w for w in warnings if "ENABLE_DOCINTEL" in w and "not a recognized bool" in w
    ]
    assert len(matching) == 1


def test_validate_env_required_only_mode(monkeypatch):
    # Clear required vars to trigger warnings
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    warnings = validate_env(required_only=True)
    required_names = [n for n, m in ENV_SCHEMA.items() if m.get("required")]
    for name in required_names:
        assert any(name in w for w in warnings)


def test_validate_env_clean_returns_empty(monkeypatch):
    # Unset all schema vars so only required-presence checks trigger
    for name in ENV_SCHEMA:
        monkeypatch.delenv(name, raising=False)
    # Set required vars to dummy values
    monkeypatch.setenv("OCR_API_KEY", "test-key")
    monkeypatch.setenv("CELERY_BROKER_URL", "amqp://localhost")
    warnings = validate_env()
    assert warnings == []


# ---------------------------------------------------------------------------
# Schema consistency
# ---------------------------------------------------------------------------


def test_env_schema_has_required_fields():
    """Every schema entry has type, default, and description."""
    for name, meta in ENV_SCHEMA.items():
        assert "type" in meta, f"{name} missing 'type'"
        assert "default" in meta, f"{name} missing 'default'"
        assert "description" in meta, f"{name} missing 'description'"


def test_env_schema_types_are_valid():
    """All type values must be one of the supported types."""
    allowed_types = {str, int, float, bool}
    for name, meta in ENV_SCHEMA.items():
        assert meta["type"] in allowed_types, f"{name} has unsupported type {meta['type']}"
