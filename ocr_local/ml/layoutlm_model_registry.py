"""LayoutLMv3 model registry for versioning fine-tuned checkpoints.

Provides a lightweight, file-based registry for tracking trained
LayoutLMv3 token-classification models.  Each registered model is
described by a :class:`ModelRegistryEntry` and stored in a central
``manifest.json`` file.

Pure Python — no ``torch``, ``transformers``, or heavy-ML imports.

Environment Variables:
    LAYOUTLM_REGISTRY_DIR (str):
        Directory for the model registry.  Default: ``"./models/registry"``.
    LAYOUTLM_ACTIVE_MODEL (str):
        ``"name:version"`` string identifying the currently active model.
        If set, :meth:`ModelRegistry.get_active_model` returns the
        matching entry.

Typical usage::

    from layoutlm_model_registry import ModelRegistry

    registry = ModelRegistry()
    entry = registry.register(
        model_dir="./models/out",
        name="forensic-v1",
        version="1.0.0",
        label_set_name="forensic",
        metrics={"f1": 0.92},
    )
    latest = registry.get_model("forensic-v1")
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_DIR = os.environ.get(
    "LAYOUTLM_REGISTRY_DIR", "./models/registry"
)

MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ModelRegistryEntry:
    """Descriptor for a registered LayoutLMv3 model checkpoint.

    Attributes:
        name:           Human-readable model name (e.g. ``"forensic-v1"``).
        version:        Semantic version string (e.g. ``"1.0.0"``).
        model_path:     Path to the saved model directory / checkpoint.
        adapter_path:   Optional path to a LoRA adapter directory.
        label_set_name: Name of the :class:`LabelSet` used for training.
        created_at:     ISO-8601 timestamp of registration.
        metrics:        Dict of evaluation metrics (e.g. ``{"f1": 0.92}``).
        description:    Free-text description of the model.
    """

    name: str
    version: str
    model_path: str
    adapter_path: Optional[str] = None
    label_set_name: str = "default"
    created_at: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ResolvedModelSelection:
    """Resolved live-inference model selection."""

    model_path: str
    source: str
    active_model_spec: str = ""
    registry_name: str = ""
    registry_version: str = ""
    adapter_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class ModelRegistry:
    """File-based registry for LayoutLMv3 fine-tuned model checkpoints.

    Models are tracked in a ``manifest.json`` inside *registry_dir*.
    Each entry records the model name, version, path, label set, and
    optional evaluation metrics.
    """

    def __init__(self, registry_dir: Optional[str] = None):
        """Initialize the registry.

        Args:
            registry_dir: Directory for the registry.  Defaults to the
                ``LAYOUTLM_REGISTRY_DIR`` env var or ``"./models/registry"``.
        """
        if registry_dir is not None:
            self._dir = Path(registry_dir)
        else:
            self._dir = Path(DEFAULT_REGISTRY_DIR)
        self._manifest_path = self._dir / MANIFEST_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_dir: str,
        name: str,
        version: str,
        label_set_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        description: str = "",
        adapter_path: Optional[str] = None,
    ) -> ModelRegistryEntry:
        """Register a trained model in the registry.

        If *model_dir* exists, its path is recorded.  The registry
        directory and manifest are created if they do not exist.

        Args:
            model_dir:      Path to the saved model checkpoint directory.
            name:           A unique name for this model.
            version:        Semantic version string.
            label_set_name: Name of the label set used during training.
            metrics:        Optional evaluation metrics dict.
            description:    Optional free-text description.
            adapter_path:   Optional path to LoRA adapter weights.

        Returns:
            The newly created :class:`ModelRegistryEntry`.

        Raises:
            ValueError: If a model with the same *name* and *version*
                        is already registered.
        """
        entries = self._load_manifest()

        # Check for duplicate
        for entry in entries:
            if entry.name == name and entry.version == version:
                raise ValueError(
                    f"Model {name!r} version {version!r} is already "
                    f"registered. Use a different version string."
                )

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_entry = ModelRegistryEntry(
            name=name,
            version=version,
            model_path=str(Path(model_dir).resolve()),
            adapter_path=(
                str(Path(adapter_path).resolve()) if adapter_path else None
            ),
            label_set_name=label_set_name,
            created_at=now,
            metrics=metrics or {},
            description=description,
        )

        entries.append(new_entry)
        self._save_manifest(entries)

        logger.info(
            "Registered model %r v%s at %s",
            name, version, new_entry.model_path,
        )
        return new_entry

    def list_models(self) -> List[ModelRegistryEntry]:
        """Return all registered models.

        Returns:
            List of :class:`ModelRegistryEntry` instances, ordered by
            registration time (oldest first).
        """
        return self._load_manifest()

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> ModelRegistryEntry:
        """Retrieve a registered model by name and optional version.

        If *version* is omitted, the latest (most recently registered)
        entry with the given *name* is returned.

        Args:
            name:    Model name to look up.
            version: Specific version string.  If ``None``, returns the
                     latest version.

        Returns:
            The matching :class:`ModelRegistryEntry`.

        Raises:
            KeyError: If no model with the given *name* (and *version*)
                      is found.
        """
        entries = self._load_manifest()
        matches = [e for e in entries if e.name == name]
        if not matches:
            raise KeyError(f"No model registered with name {name!r}")

        if version is not None:
            versioned = [e for e in matches if e.version == version]
            if not versioned:
                raise KeyError(
                    f"No model {name!r} with version {version!r}. "
                    f"Available: {[e.version for e in matches]}"
                )
            return versioned[-1]

        # Return latest (last registered)
        return matches[-1]

    def get_active_model(self) -> Optional[ModelRegistryEntry]:
        """Return the currently active model based on environment.

        Reads ``LAYOUTLM_ACTIVE_MODEL`` env var, which should be in
        ``"name:version"`` format (e.g. ``"forensic-v1:1.0.0"``).
        If only a name is given (no colon), the latest version is used.

        Returns:
            The matching :class:`ModelRegistryEntry`, or ``None`` if the
            env var is unset or the model is not found.
        """
        active_spec = os.environ.get("LAYOUTLM_ACTIVE_MODEL", "")
        if not active_spec:
            return None

        if ":" in active_spec:
            name, version = active_spec.rsplit(":", 1)
        else:
            name = active_spec
            version = None

        try:
            return self.get_model(name, version)
        except KeyError:
            logger.warning(
                "LAYOUTLM_ACTIVE_MODEL=%r not found in registry", active_spec,
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> List[ModelRegistryEntry]:
        """Load the manifest from disk, returning an empty list if absent."""
        if not self._manifest_path.is_file():
            return []

        with open(self._manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)

        entries: List[ModelRegistryEntry] = []
        for item in data.get("models", []):
            entries.append(
                ModelRegistryEntry(
                    name=item.get("name", ""),
                    version=item.get("version", ""),
                    model_path=item.get("model_path", ""),
                    adapter_path=item.get("adapter_path"),
                    label_set_name=item.get("label_set_name", "default"),
                    created_at=item.get("created_at", ""),
                    metrics=item.get("metrics", {}),
                    description=item.get("description", ""),
                )
            )
        return entries

    def _save_manifest(self, entries: List[ModelRegistryEntry]) -> None:
        """Persist the manifest to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "1.0",
            "models": [asdict(e) for e in entries],
        }
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)


def resolve_active_model_selection(
    fallback_model_path: str,
    registry_dir: Optional[str] = None,
) -> ResolvedModelSelection:
    """Resolve the model path for live inference.

    Prefers the active registry entry when ``LAYOUTLM_ACTIVE_MODEL`` is set and
    resolvable. Falls back to the provided model path when no active entry is
    configured or the entry cannot be resolved.
    """
    try:
        registry = ModelRegistry(registry_dir=registry_dir)
        active_entry = registry.get_active_model()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Failed to resolve active LayoutLM model from registry: %s. "
            "Falling back to configured model path.",
            exc,
        )
        return ResolvedModelSelection(
            model_path=fallback_model_path,
            source="fallback",
        )

    if active_entry is None or not active_entry.model_path.strip():
        return ResolvedModelSelection(
            model_path=fallback_model_path,
            source="fallback",
        )

    active_model_path = Path(active_entry.model_path)
    if not active_model_path.exists():
        logger.warning(
            "Active LayoutLM model path %s does not exist. Falling back to "
            "configured model path.",
            active_entry.model_path,
        )
        return ResolvedModelSelection(
            model_path=fallback_model_path,
            source="fallback",
        )

    return ResolvedModelSelection(
        model_path=str(active_model_path),
        source="registry",
        active_model_spec=f"{active_entry.name}:{active_entry.version}",
        registry_name=active_entry.name,
        registry_version=active_entry.version,
        adapter_path=active_entry.adapter_path,
    )
