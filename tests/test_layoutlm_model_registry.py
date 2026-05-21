"""Tests for layoutlm_model_registry — Model versioning for LayoutLMv3.

Covers ModelRegistryEntry, ModelRegistry (register, list, get, active),
manifest persistence, and error cases. All tests use tmp_path and
require NO torch/transformers.

Run with: python -m pytest tests/test_layoutlm_model_registry.py -v
"""

import json

import pytest

from layoutlm_model_registry import (
    MANIFEST_FILENAME,
    ModelRegistry,
    ModelRegistryEntry,
    ResolvedModelSelection,
    resolve_active_model_selection,
)

# ---------------------------------------------------------------------------
# ModelRegistryEntry tests
# ---------------------------------------------------------------------------


class TestModelRegistryEntry:
    """Tests for the ModelRegistryEntry dataclass."""

    def test_default_values(self):
        """Required fields work, optionals have defaults."""
        entry = ModelRegistryEntry(
            name="test-model",
            version="1.0.0",
            model_path="/models/test",
        )
        assert entry.name == "test-model"
        assert entry.version == "1.0.0"
        assert entry.adapter_path is None
        assert entry.label_set_name == "default"
        assert entry.created_at == ""
        assert entry.metrics == {}
        assert entry.description == ""

    def test_full_values(self):
        """All fields are set correctly."""
        entry = ModelRegistryEntry(
            name="forensic-v1",
            version="2.0.0",
            model_path="/models/forensic",
            adapter_path="/adapters/forensic",
            label_set_name="forensic",
            created_at="2024-01-01T00:00:00+00:00",
            metrics={"f1": 0.92, "precision": 0.90},
            description="Fine-tuned on forensic docs",
        )
        assert entry.label_set_name == "forensic"
        assert entry.metrics["f1"] == 0.92
        assert entry.adapter_path == "/adapters/forensic"


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_empty_registry(self, tmp_path):
        """A fresh registry lists no models."""
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        assert reg.list_models() == []

    def test_register_model(self, tmp_path):
        """Registering a model returns a valid entry."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        entry = reg.register(
            model_dir=str(model_dir),
            name="test-model",
            version="1.0.0",
            label_set_name="default",
            metrics={"f1": 0.88},
            description="Test model",
        )

        assert entry.name == "test-model"
        assert entry.version == "1.0.0"
        assert entry.label_set_name == "default"
        assert entry.metrics == {"f1": 0.88}
        assert entry.created_at != ""

    def test_list_models_after_register(self, tmp_path):
        """list_models returns all registered entries."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        reg.register(str(model_dir), "m1", "1.0", "default")
        reg.register(str(model_dir), "m2", "1.0", "forensic")

        models = reg.list_models()
        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"m1", "m2"}

    def test_duplicate_version_raises(self, tmp_path):
        """Registering the same name+version twice raises ValueError."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        reg.register(str(model_dir), "m1", "1.0", "default")
        with pytest.raises(ValueError, match="already registered"):
            reg.register(str(model_dir), "m1", "1.0", "default")

    def test_get_model_by_name(self, tmp_path):
        """get_model returns the latest version when version is omitted."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        reg.register(str(model_dir), "m1", "1.0", "default")
        reg.register(str(model_dir), "m1", "2.0", "default")

        entry = reg.get_model("m1")
        assert entry.version == "2.0"

    def test_get_model_by_version(self, tmp_path):
        """get_model returns the exact version requested."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        reg.register(str(model_dir), "m1", "1.0", "default")
        reg.register(str(model_dir), "m1", "2.0", "default")

        entry = reg.get_model("m1", version="1.0")
        assert entry.version == "1.0"

    def test_get_model_not_found(self, tmp_path):
        """KeyError when model name is not in the registry."""
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        with pytest.raises(KeyError, match="No model registered"):
            reg.get_model("nonexistent")

    def test_get_model_version_not_found(self, tmp_path):
        """KeyError when version doesn't exist for a known model."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        reg.register(str(model_dir), "m1", "1.0", "default")
        with pytest.raises(KeyError, match="No model"):
            reg.get_model("m1", version="99.0")

    def test_manifest_persistence(self, tmp_path):
        """Manifest is persisted to disk and survives re-instantiation."""
        reg_dir = str(tmp_path / "reg")
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()

        reg1 = ModelRegistry(registry_dir=reg_dir)
        reg1.register(str(model_dir), "m1", "1.0", "default")

        # New registry instance reads from disk
        reg2 = ModelRegistry(registry_dir=reg_dir)
        models = reg2.list_models()
        assert len(models) == 1
        assert models[0].name == "m1"

    def test_manifest_is_valid_json(self, tmp_path):
        """The manifest file is valid JSON with expected schema."""
        reg_dir = tmp_path / "reg"
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(reg_dir))

        reg.register(str(model_dir), "m1", "1.0", "default", metrics={"f1": 0.9})

        manifest_path = reg_dir / MANIFEST_FILENAME
        assert manifest_path.is_file()

        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["schema_version"] == "1.0"
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "m1"
        assert data["models"][0]["metrics"]["f1"] == 0.9


# ---------------------------------------------------------------------------
# get_active_model tests
# ---------------------------------------------------------------------------


class TestGetActiveModel:
    """Tests for get_active_model env-var resolution."""

    def test_no_env_returns_none(self, tmp_path, monkeypatch):
        """Without LAYOUTLM_ACTIVE_MODEL, returns None."""
        monkeypatch.delenv("LAYOUTLM_ACTIVE_MODEL", raising=False)
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        assert reg.get_active_model() is None

    def test_active_model_with_version(self, tmp_path, monkeypatch):
        """name:version format resolves correctly."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        reg.register(str(model_dir), "m1", "1.0", "default")

        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "m1:1.0")
        entry = reg.get_active_model()
        assert entry is not None
        assert entry.name == "m1"
        assert entry.version == "1.0"

    def test_active_model_name_only(self, tmp_path, monkeypatch):
        """Name without version returns latest."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        reg.register(str(model_dir), "m1", "1.0", "default")
        reg.register(str(model_dir), "m1", "2.0", "default")

        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "m1")
        entry = reg.get_active_model()
        assert entry is not None
        assert entry.version == "2.0"

    def test_active_model_not_found(self, tmp_path, monkeypatch):
        """Unknown active model returns None (with logged warning)."""
        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "nonexistent:1.0")
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        assert reg.get_active_model() is None

    def test_register_with_adapter_path(self, tmp_path):
        """Adapter path is stored correctly."""
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))

        entry = reg.register(
            str(model_dir), "m1", "1.0", "default",
            adapter_path=str(adapter_dir),
        )
        assert entry.adapter_path is not None
        assert "adapters" in entry.adapter_path


class TestResolveActiveModelSelection:
    """Tests for live inference model resolution."""

    def test_falls_back_without_active_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LAYOUTLM_ACTIVE_MODEL", raising=False)
        selection = resolve_active_model_selection(
            fallback_model_path="/models/fallback",
            registry_dir=str(tmp_path / "reg"),
        )
        assert isinstance(selection, ResolvedModelSelection)
        assert selection.model_path == "/models/fallback"
        assert selection.source == "fallback"
        assert selection.active_model_spec == ""

    def test_uses_registry_active_model(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "model_out"
        model_dir.mkdir()
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        reg.register(str(model_dir), "m1", "1.0", "default")
        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "m1:1.0")

        selection = resolve_active_model_selection(
            fallback_model_path="/models/fallback",
            registry_dir=str(tmp_path / "reg"),
        )
        assert selection.source == "registry"
        assert selection.model_path == str(model_dir.resolve())
        assert selection.active_model_spec == "m1:1.0"
        assert selection.registry_name == "m1"
        assert selection.registry_version == "1.0"

    def test_invalid_active_model_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "missing:1.0")
        selection = resolve_active_model_selection(
            fallback_model_path="/models/fallback",
            registry_dir=str(tmp_path / "reg"),
        )
        assert selection.source == "fallback"
        assert selection.model_path == "/models/fallback"

    def test_missing_checkpoint_falls_back(self, tmp_path, monkeypatch):
        reg = ModelRegistry(registry_dir=str(tmp_path / "reg"))
        reg.register(str(tmp_path / "missing-model"), "m1", "1.0", "default")
        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "m1:1.0")

        selection = resolve_active_model_selection(
            fallback_model_path="/models/fallback",
            registry_dir=str(tmp_path / "reg"),
        )
        assert selection.source == "fallback"
        assert selection.model_path == "/models/fallback"

    def test_corrupt_manifest_falls_back(self, tmp_path, monkeypatch):
        reg_dir = tmp_path / "reg"
        reg_dir.mkdir()
        manifest_path = reg_dir / MANIFEST_FILENAME
        manifest_path.write_text("{not-json", encoding="utf-8")
        monkeypatch.setenv("LAYOUTLM_ACTIVE_MODEL", "m1:1.0")

        selection = resolve_active_model_selection(
            fallback_model_path="/models/fallback",
            registry_dir=str(reg_dir),
        )
        assert selection.source == "fallback"
        assert selection.model_path == "/models/fallback"
