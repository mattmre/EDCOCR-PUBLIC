"""Tests for feature_flags.py runtime feature registry."""

from __future__ import annotations

from unittest import mock

import feature_flags


def test_get_pipeline_features_returns_dict():
    """get_pipeline_features() returns a dict of str -> bool."""
    features = feature_flags.get_pipeline_features(force_refresh=True)
    assert isinstance(features, dict)
    assert len(features) > 0
    for key, value in features.items():
        assert isinstance(key, str), f"Key {key!r} is not a string"
        assert isinstance(value, bool), f"Value for {key!r} is not a bool"


def test_is_feature_available_unknown_returns_false():
    """Unknown feature names return False, not KeyError."""
    assert feature_flags.is_feature_available("nonexistent_feature_xyz") is False


def test_feature_flags_cached():
    """Results are cached -- second call returns same object without re-probing."""
    # Clear cache
    feature_flags._cache = None
    feature_flags.get_pipeline_features()
    # Mutate the internal cache to detect if a second call re-probes
    feature_flags._cache["_sentinel_test_key"] = True  # type: ignore[index]
    second = feature_flags.get_pipeline_features()
    # The sentinel should appear in the second call (it reads from cache)
    assert "_sentinel_test_key" in second
    # Clean up
    feature_flags._cache = None


def test_force_refresh_clears_cache():
    """force_refresh=True re-probes all features."""
    # Populate cache
    feature_flags.get_pipeline_features(force_refresh=True)
    # Inject a sentinel
    feature_flags._cache["_sentinel_refresh_test"] = True  # type: ignore[index]
    # Force refresh should discard the sentinel
    refreshed = feature_flags.get_pipeline_features(force_refresh=True)
    assert "_sentinel_refresh_test" not in refreshed
    # Clean up
    feature_flags._cache = None


def test_feature_map_covers_known_modules():
    """Spot-check that key features are in the feature map."""
    expected = {"ner", "handwriting", "classification", "custody", "validation"}
    actual_keys = set(feature_flags._FEATURE_MAP.keys())
    assert expected.issubset(actual_keys)


def test_import_error_returns_false():
    """When a module cannot be imported, the feature is marked False."""
    with mock.patch.dict(
        feature_flags._FEATURE_MAP,
        {"_test_missing": ("_nonexistent_module_xyz_123", None)},
    ):
        features = feature_flags.get_pipeline_features(force_refresh=True)
        assert features["_test_missing"] is False
    # Clean up
    feature_flags._cache = None


def test_returns_copy_not_reference():
    """get_pipeline_features returns a copy, so mutations do not affect cache."""
    features = feature_flags.get_pipeline_features(force_refresh=True)
    features["_mutation_test"] = True
    second = feature_flags.get_pipeline_features()
    assert "_mutation_test" not in second
    # Clean up
    feature_flags._cache = None
