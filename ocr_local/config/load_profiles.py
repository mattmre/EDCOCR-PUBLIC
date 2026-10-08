"""ES-8 — Tunable load profiles mapped to hardware + deployment profiles.

Load profiles describe *how hard* to exercise the scale bed (corpus size,
concurrency, wall/RSS caps, feature matrix flags). They compose with:

  - ES-1 hardware auto-tune (host-fit knobs)
  - ES-4 deployment profiles (topology class knobs)

Selection (env):
  OCR_LOAD_PROFILE=light|medium|heavy|stress

This module does **not** claim multi-host farm readiness (ES-9…ES-12) or
full GPU 5k OCR soak. It maps load intensity to host-fit and topology
templates so operators can pick a coherent plan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

LOAD_PROFILE_ENV = "OCR_LOAD_PROFILE"
KNOWN_LOAD_PROFILES = ("light", "medium", "heavy", "stress")


@dataclass(frozen=True)
class LoadProfile:
    """Named load intensity template for scale / feature matrix beds."""

    name: str
    corpus_size: int  # ES-5 tier nSIZE
    max_docs: int  # 0 = all primary docs in tier
    workers: int  # harness concurrency (ingest/feature matrix)
    max_wall_seconds: float
    max_rss_mb: int
    enable_features: tuple[str, ...]  # validation/custody/classification
    preferred_deployment_profiles: tuple[str, ...]  # ES-4 names, preference order
    min_cpu: int
    min_ram_mb: int
    prefer_gpu: bool
    description: str = ""
    honest_limits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["enable_features"] = list(self.enable_features)
        d["preferred_deployment_profiles"] = list(self.preferred_deployment_profiles)
        d["honest_limits"] = list(self.honest_limits)
        return d


def _builtins() -> dict[str, LoadProfile]:
    common_limit = (
        "Load profile is a plan template — not multi-host farm proof (ES-9+).",
        "Default soak path remains ingest/feature modules, not full GPU OCR 5k.",
    )
    return {
        "light": LoadProfile(
            name="light",
            corpus_size=100,
            max_docs=0,
            workers=2,
            max_wall_seconds=120.0,
            max_rss_mb=2048,
            enable_features=("validation", "custody"),
            preferred_deployment_profiles=("sidecar", "solo"),
            min_cpu=2,
            min_ram_mb=2048,
            prefer_gpu=False,
            description="Smoke-scale load: n100, lean features, sidecar/solo fit",
            honest_limits=common_limit,
        ),
        "medium": LoadProfile(
            name="medium",
            corpus_size=1000,
            max_docs=0,
            workers=4,
            max_wall_seconds=300.0,
            max_rss_mb=4096,
            enable_features=("validation", "custody", "classification"),
            preferred_deployment_profiles=("solo", "small-farm"),
            min_cpu=4,
            min_ram_mb=4096,
            prefer_gpu=False,
            description="Standard scale bed: n1000 + full feature matrix",
            honest_limits=common_limit,
        ),
        "heavy": LoadProfile(
            name="heavy",
            corpus_size=2500,
            max_docs=0,
            workers=8,
            max_wall_seconds=600.0,
            max_rss_mb=8192,
            enable_features=("validation", "custody", "classification"),
            preferred_deployment_profiles=("solo", "small-farm", "large-farm"),
            min_cpu=8,
            min_ram_mb=8192,
            prefer_gpu=True,
            description="Large single-host bed: n2500, higher concurrency",
            honest_limits=common_limit
            + ("Prefer GPU host when available for future OCR mode.",),
        ),
        "stress": LoadProfile(
            name="stress",
            corpus_size=5000,
            max_docs=0,
            workers=12,
            max_wall_seconds=1200.0,
            max_rss_mb=16384,
            enable_features=("validation", "custody", "classification"),
            preferred_deployment_profiles=("large-farm", "small-farm", "solo"),
            min_cpu=12,
            min_ram_mb=16384,
            prefer_gpu=True,
            description="Ceiling single-host bed: n5000 stress envelope",
            honest_limits=common_limit
            + ("large-farm knobs are templates only — not multi-host certification.",),
        ),
    }


def get_builtin_load_profile(name: str) -> LoadProfile | None:
    return _builtins().get(name.strip().lower())


def list_load_profiles() -> list[str]:
    return list(KNOWN_LOAD_PROFILES)


def resolve_load_profile(
    env: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
) -> LoadProfile | None:
    """Resolve load profile from explicit name or OCR_LOAD_PROFILE env."""
    e = env if env is not None else os.environ
    if name:
        return get_builtin_load_profile(name)
    raw = str(e.get(LOAD_PROFILE_ENV, "") or "").strip().lower()
    if not raw:
        return None
    return get_builtin_load_profile(raw)


def select_deployment_for_load(
    load: LoadProfile,
    *,
    available: tuple[str, ...] | None = None,
) -> str:
    """Pick first preferred ES-4 deployment profile that is available."""
    avail = set(available or ("solo", "sidecar", "small-farm", "large-farm"))
    for pref in load.preferred_deployment_profiles:
        if pref in avail:
            return pref
    return "solo"


def host_fits_load(load: LoadProfile, hardware: Mapping[str, Any]) -> dict[str, Any]:
    """Check whether detected host meets load minimums (soft guidance)."""
    cpu = int(hardware.get("cpu_count") or 0)
    ram = int(hardware.get("ram_mb") or 0)
    has_cuda = bool(hardware.get("has_cuda"))
    cpu_ok = cpu >= load.min_cpu
    ram_ok = ram >= load.min_ram_mb
    gpu_ok = (not load.prefer_gpu) or has_cuda
    fits = cpu_ok and ram_ok  # GPU is preference, not hard fail
    return {
        "fits": fits,
        "cpu_ok": cpu_ok,
        "ram_ok": ram_ok,
        "gpu_preference_met": gpu_ok,
        "cpu_count": cpu,
        "ram_mb": ram,
        "has_cuda": has_cuda,
        "min_cpu": load.min_cpu,
        "min_ram_mb": load.min_ram_mb,
        "prefer_gpu": load.prefer_gpu,
        "notes": []
        if fits
        else [
            f"host may be undersized for load={load.name} "
            f"(need cpu>={load.min_cpu} ram_mb>={load.min_ram_mb})"
        ],
    }


def build_load_hardware_plan(
    load_name: str,
    *,
    env: Mapping[str, str] | None = None,
    hardware: Any | None = None,
    hw_profile: Any | None = None,
) -> dict[str, Any]:
    """Compose load + ES-1 hardware recommend + ES-4 deployment selection.

    Returns a plan dict suitable for operators and ES-8 evidence.
    """
    from ocr_local.config.deployment_profiles import (  # noqa: WPS433
        get_builtin_profile,
        list_profiles,
    )
    from ocr_local.infra.hardware_autotune import (  # noqa: WPS433
        detect_host_hardware,
        recommend_pipeline_profile,
    )

    load = get_builtin_load_profile(load_name)
    if load is None:
        raise ValueError(f"unknown load profile: {load_name!r}")

    hw = hardware if hardware is not None else detect_host_hardware()
    if hasattr(hw, "__dataclass_fields__"):
        hw_dict = asdict(hw)
    elif isinstance(hw, Mapping):
        hw_dict = dict(hw)
    else:
        hw_dict = {
            "cpu_count": getattr(hw, "cpu_count", 0),
            "ram_mb": getattr(hw, "ram_mb", 0),
            "gpu_count": getattr(hw, "gpu_count", 0),
            "has_cuda": getattr(hw, "has_cuda", False),
            "vram_total_mb": getattr(hw, "vram_total_mb", 0),
        }

    tune = hw_profile if hw_profile is not None else recommend_pipeline_profile(hw)
    tune_dict = tune.to_dict() if hasattr(tune, "to_dict") else dict(tune)

    dep_name = select_deployment_for_load(load, available=tuple(list_profiles()))
    dep = get_builtin_profile(dep_name)
    dep_dict = dep.to_dict() if dep is not None else {"name": dep_name}

    fit = host_fits_load(load, hw_dict)

    # Combined knobs: start from deployment, overlay hardware recommend for
    # worker/queue intensity when host is larger; load profile sets soak workers.
    combined_knobs = {
        "num_extractors": int(
            tune_dict.get("num_extractors") or dep.num_extractors if dep else 4
        ),
        "num_workers": int(
            tune_dict.get("num_workers") or dep.num_workers if dep else 4
        ),
        "num_compressors": int(
            tune_dict.get("num_compressors") or (dep.num_compressors if dep else 2)
        ),
        "image_queue_size": int(
            tune_dict.get("image_queue_size") or (dep.image_queue_size if dep else 25)
        ),
        "chunk_queue_size": int(
            tune_dict.get("chunk_queue_size") or (dep.chunk_queue_size if dep else 12)
        ),
        "dpi": int(tune_dict.get("dpi") or (dep.dpi if dep else 300)),
        "soak_workers": load.workers,
        "corpus_size": load.corpus_size,
        "max_docs": load.max_docs,
        "max_wall_seconds": load.max_wall_seconds,
        "max_rss_mb": load.max_rss_mb,
        "enable_features": list(load.enable_features),
    }

    return {
        "load_profile": load.to_dict(),
        "hardware": hw_dict,
        "hardware_tune": tune_dict,
        "deployment_profile": dep_dict,
        "deployment_selected": dep_name,
        "host_fit": fit,
        "combined_knobs": combined_knobs,
        "mapping_rules": [
            "Load profile sets corpus_size, soak workers, caps, feature set.",
            "ES-1 recommend_pipeline_profile supplies host-fit pipeline knobs.",
            "ES-4 deployment profile selected from load preferred list + available.",
            "Explicit NUM_* env still wins at pipeline_config create time (ES-2/ES-4).",
        ],
        "honest_limits": list(load.honest_limits),
    }
