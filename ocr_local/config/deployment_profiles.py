"""ES-4 — Modular deployment profiles (solo / sidecar / small-farm / large-farm).

Named knob templates for topology class. These are **config surfaces**, not
claims of multi-host farm readiness (farm proof is ES-9…ES-12).

Selection (env):
  OCR_DEPLOYMENT_PROFILE=solo|sidecar|small-farm|large-farm
  OCR_PROFILE=…                 (alias)
  OCR_PROFILE_FILE=/path.json   (optional overlay; must include name + knobs)

Precedence for each knob field:
  1. Explicit NUM_* / IMAGE_QUEUE_SIZE / DPI env (always wins)
  2. OCR_AUTO_TUNE hardware recommend (when enabled)
  3. Deployment profile defaults
  4. PipelineConfig stock defaults

Profile application only fills knobs the operator did not set explicitly.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROFILE_ENV_KEYS = ("OCR_DEPLOYMENT_PROFILE", "OCR_PROFILE")
PROFILE_FILE_ENV = "OCR_PROFILE_FILE"

# Explicit env that blocks profile fill for that field (same contract as ES-2).
_EXPLICIT_KNOB_ENV: dict[str, str] = {
    "num_extractors": "NUM_EXTRACTORS",
    "num_workers": "NUM_WORKERS",
    "num_compressors": "NUM_COMPRESSORS",
    "image_queue_size": "IMAGE_QUEUE_SIZE",
    "chunk_queue_size": "CHUNK_QUEUE_SIZE",
    "dpi": "DPI",
}

KNOWN_PROFILES = ("solo", "sidecar", "small-farm", "large-farm")


@dataclass(frozen=True)
class DeploymentProfile:
    """Named topology knob template."""

    name: str
    num_extractors: int
    num_workers: int
    num_compressors: int
    image_queue_size: int
    chunk_queue_size: int
    dpi: int
    description: str = ""
    topology_class: str = ""  # solo | sidecar | farm
    honest_limits: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["honest_limits"] = list(self.honest_limits)
        return d

    def knob_dict(self) -> dict[str, int]:
        return {
            "num_extractors": self.num_extractors,
            "num_workers": self.num_workers,
            "num_compressors": self.num_compressors,
            "image_queue_size": self.image_queue_size,
            "chunk_queue_size": self.chunk_queue_size,
            "dpi": self.dpi,
        }


def _builtin_profiles() -> dict[str, DeploymentProfile]:
    farm_limit = ("Knob template only — does not prove multi-host farm (ES-9…ES-12).",)
    return {
        "solo": DeploymentProfile(
            name="solo",
            num_extractors=4,
            num_workers=4,
            num_compressors=2,
            image_queue_size=25,
            chunk_queue_size=12,
            dpi=300,
            description="Single-machine SOLO defaults (matches docker-compose.solo.yml spirit)",
            topology_class="solo",
            honest_limits=(
                "Single-host. Pair with OCR_AUTO_TUNE for host-specific refine.",
            ),
        ),
        "sidecar": DeploymentProfile(
            name="sidecar",
            num_extractors=2,
            num_workers=2,
            num_compressors=1,
            image_queue_size=16,
            chunk_queue_size=8,
            dpi=300,
            description="Lean peer-app SIDECAR (EDCOMMANDER-class) defaults",
            topology_class="sidecar",
            honest_limits=(
                "Assumes shared-host or small peer footprint; not farm routing.",
            ),
        ),
        "small-farm": DeploymentProfile(
            name="small-farm",
            num_extractors=6,
            num_workers=8,
            num_compressors=4,
            image_queue_size=80,
            chunk_queue_size=30,
            dpi=300,
            description="Small multi-worker / small-site baseline knobs",
            topology_class="farm",
            honest_limits=farm_limit,
        ),
        "large-farm": DeploymentProfile(
            name="large-farm",
            num_extractors=12,
            num_workers=16,
            num_compressors=8,
            image_queue_size=160,
            chunk_queue_size=50,
            dpi=300,
            description="Large multi-worker baseline knobs (single-process ceilings)",
            topology_class="farm",
            honest_limits=farm_limit
            + (
                "Per-process caps; true multi-host needs ES-10/11 registration+routing.",
            ),
        ),
    }


def list_profiles() -> list[str]:
    return list(KNOWN_PROFILES)


def get_builtin_profile(name: str) -> DeploymentProfile | None:
    key = name.strip().lower().replace("_", "-")
    return _builtin_profiles().get(key)


def load_profile_file(path: Path) -> DeploymentProfile:
    """Load a JSON profile file with name + knob fields."""
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name") or path.stem).strip().lower().replace("_", "-")
    base = get_builtin_profile(name)
    knobs = {
        "num_extractors": int(
            data.get("num_extractors", base.num_extractors if base else 4)
        ),
        "num_workers": int(data.get("num_workers", base.num_workers if base else 4)),
        "num_compressors": int(
            data.get("num_compressors", base.num_compressors if base else 2)
        ),
        "image_queue_size": int(
            data.get("image_queue_size", base.image_queue_size if base else 25)
        ),
        "chunk_queue_size": int(
            data.get("chunk_queue_size", base.chunk_queue_size if base else 12)
        ),
        "dpi": int(data.get("dpi", base.dpi if base else 300)),
    }
    return DeploymentProfile(
        name=name,
        description=str(data.get("description", base.description if base else "")),
        topology_class=str(
            data.get("topology_class", base.topology_class if base else "custom")
        ),
        honest_limits=tuple(
            data.get("honest_limits")
            or (base.honest_limits if base else ("Custom file profile.",))
        ),
        extra={
            k: v
            for k, v in data.items()
            if k
            not in {"name", *knobs, "description", "topology_class", "honest_limits"}
        },
        **knobs,
    )


def resolve_profile_name(env: Mapping[str, str] | None = None) -> str | None:
    e = env if env is not None else os.environ
    for key in PROFILE_ENV_KEYS:
        raw = e.get(key)
        if raw and str(raw).strip():
            return str(raw).strip().lower().replace("_", "-")
    return None


def resolve_deployment_profile(
    env: Mapping[str, str] | None = None,
) -> DeploymentProfile | None:
    """Resolve profile from OCR_PROFILE_FILE or OCR_DEPLOYMENT_PROFILE / OCR_PROFILE."""
    e = env if env is not None else os.environ
    file_raw = e.get(PROFILE_FILE_ENV)
    if file_raw and str(file_raw).strip():
        path = Path(str(file_raw).strip())
        if not path.is_file():
            raise FileNotFoundError(f"{PROFILE_FILE_ENV} not found: {path}")
        return load_profile_file(path)
    name = resolve_profile_name(e)
    if not name:
        return None
    prof = get_builtin_profile(name)
    if prof is None:
        raise ValueError(
            f"Unknown deployment profile {name!r}; known: {', '.join(KNOWN_PROFILES)}"
        )
    return prof


def apply_deployment_profile_to_pipeline_config(
    cfg: Any,
    profile: DeploymentProfile,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Fill unset knobs from *profile*; explicit NUM_* env wins."""
    import dataclasses

    e = env if env is not None else os.environ
    updates: dict[str, Any] = {}
    for field_name, value in profile.knob_dict().items():
        env_key = _EXPLICIT_KNOB_ENV[field_name]
        if env_key in e:
            continue
        if hasattr(cfg, field_name):
            updates[field_name] = value
    if not updates:
        return cfg
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.replace(cfg, **updates)
    for k, v in updates.items():
        setattr(cfg, k, v)
    return cfg


def maybe_apply_deployment_profile_to_pipeline_config(
    cfg: Any,
    env: Mapping[str, str] | None = None,
) -> tuple[Any, DeploymentProfile | None]:
    """Apply named deployment profile when env requests one."""
    e = env if env is not None else os.environ
    try:
        profile = resolve_deployment_profile(e)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Deployment profile resolve failed: %s", exc)
        return cfg, None
    if profile is None:
        return cfg, None
    applied = apply_deployment_profile_to_pipeline_config(cfg, profile, env=e)
    logger.info(
        "OCR_DEPLOYMENT_PROFILE applied %s (workers=%s extractors=%s); "
        "explicit NUM_* still take precedence; topology_class=%s",
        profile.name,
        getattr(applied, "num_workers", "?"),
        getattr(applied, "num_extractors", "?"),
        profile.topology_class,
    )
    return applied, profile


def write_builtin_profile_files(directory: Path) -> list[Path]:
    """Write JSON files for each builtin profile (operator / docs aid)."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, prof in _builtin_profiles().items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(prof.to_dict(), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
