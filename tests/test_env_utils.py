"""Tests for ocr_distributed.ocr_utils.get_env_int / get_env_float."""

import os

import pytest

from ocr_distributed.ocr_utils import get_env_float, get_env_int

# ---------------------------------------------------------------------------
# get_env_int
# ---------------------------------------------------------------------------


class TestGetEnvInt:
    """Tests for get_env_int."""

    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "42")
        assert get_env_int("TEST_INT_VAR", 10) == 42

    def test_default_when_missing(self):
        # Use a key that definitely does not exist
        key = "_MP086_MISSING_INT_VAR_UNIQUE_"
        assert os.environ.get(key) is None
        assert get_env_int(key, 99) == 99

    def test_bad_value_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
        assert get_env_int("TEST_INT_VAR", 7) == 7

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "")
        assert get_env_int("TEST_INT_VAR", 5) == 5

    def test_min_val_clamp(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "0")
        assert get_env_int("TEST_INT_VAR", 10, min_val=5) == 5

    def test_max_val_clamp(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "999")
        assert get_env_int("TEST_INT_VAR", 10, max_val=100) == 100

    def test_both_bounds(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "50")
        assert get_env_int("TEST_INT_VAR", 10, min_val=1, max_val=100) == 50

    def test_value_below_min(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "-5")
        assert get_env_int("TEST_INT_VAR", 10, min_val=0) == 0

    def test_value_above_max(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "200")
        assert get_env_int("TEST_INT_VAR", 10, max_val=100) == 100

    def test_no_bounds(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "-999")
        assert get_env_int("TEST_INT_VAR", 10) == -999

    def test_negative_default(self):
        key = "_MP086_MISSING_NEG_INT_"
        assert get_env_int(key, -1) == -1

    def test_float_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "3.14")
        assert get_env_int("TEST_INT_VAR", 7) == 7

    def test_warns_on_bad_value(self, monkeypatch, caplog):
        monkeypatch.setenv("TEST_INT_VAR", "xyz")
        result = get_env_int("TEST_INT_VAR", 42)
        assert result == 42
        assert any("Invalid" in r.message and "TEST_INT_VAR" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_env_float
# ---------------------------------------------------------------------------


class TestGetEnvFloat:
    """Tests for get_env_float."""

    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "3.14")
        assert get_env_float("TEST_FLOAT_VAR", 1.0) == pytest.approx(3.14)

    def test_default_when_missing(self):
        key = "_MP086_MISSING_FLOAT_VAR_UNIQUE_"
        assert os.environ.get(key) is None
        assert get_env_float(key, 2.5) == pytest.approx(2.5)

    def test_bad_value_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "not_a_float")
        assert get_env_float("TEST_FLOAT_VAR", 1.5) == pytest.approx(1.5)

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "")
        assert get_env_float("TEST_FLOAT_VAR", 9.9) == pytest.approx(9.9)

    def test_min_val_clamp(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "-1.0")
        assert get_env_float("TEST_FLOAT_VAR", 0.5, min_val=0.0) == pytest.approx(0.0)

    def test_max_val_clamp(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "999.9")
        assert get_env_float("TEST_FLOAT_VAR", 1.0, max_val=100.0) == pytest.approx(100.0)

    def test_both_bounds(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "50.5")
        assert get_env_float("TEST_FLOAT_VAR", 1.0, min_val=0.0, max_val=100.0) == pytest.approx(50.5)

    def test_no_bounds(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "-999.9")
        assert get_env_float("TEST_FLOAT_VAR", 1.0) == pytest.approx(-999.9)

    def test_integer_string_accepted(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "42")
        assert get_env_float("TEST_FLOAT_VAR", 1.0) == pytest.approx(42.0)

    def test_warns_on_bad_value(self, monkeypatch, caplog):
        monkeypatch.setenv("TEST_FLOAT_VAR", "abc")
        result = get_env_float("TEST_FLOAT_VAR", 1.0)
        assert result == pytest.approx(1.0)
        assert any("Invalid" in r.message and "TEST_FLOAT_VAR" in r.message for r in caplog.records)

    def test_scientific_notation(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "1e3")
        assert get_env_float("TEST_FLOAT_VAR", 1.0) == pytest.approx(1000.0)

    def test_bad_value_with_bounds_uses_default_then_clamps(self, monkeypatch):
        """When the value is invalid, the default is used, then bounds are applied."""
        monkeypatch.setenv("TEST_FLOAT_VAR", "bad")
        # default=0.5 but min_val=1.0 should clamp
        assert get_env_float("TEST_FLOAT_VAR", 0.5, min_val=1.0) == pytest.approx(1.0)
