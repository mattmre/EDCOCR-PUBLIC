"""Hardware auto-tune for SOLO / pipeline knobs (ES-1).

Detects CPU, RAM, and GPU/VRAM (when available) and recommends pipeline
worker/queue/DPI/batch settings. Does **not** mutate process globals unless
the caller explicitly applies the profile.

Opt-in later: ``OCR_AUTO_TUNE=true`` may apply recommendations at startup;
default is detect+recommend only (no silent production flip).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostHardware:
    """Snapshot of host resources."""

    cpu_count: int
    ram_mb: int
    gpu_count: int
    gpu_names: tuple[str, ...] = ()
    vram_total_mb: int = 0
    has_cuda: bool = False
    detection_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineTuneProfile:
    """Recommended pipeline knobs for SOLO / single-host runs."""

    num_extractors: int
    num_workers: int
    num_compressors: int
    image_queue_size: int
    chunk_queue_size: int
    dpi: int
    max_batch_images: int
    profile_name: str
    rationale: tuple[str, ...] = ()
    hardware: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_host_hardware() -> HostHardware:
    """Detect CPU/RAM/GPU resources without requiring CUDA at import time."""
    notes: list[str] = []
    cpu = os.cpu_count() or 1
    notes.append(f"cpu_count={cpu}")

    ram_mb = 0
    try:
        # Linux: /proc/meminfo
        meminfo = _read_meminfo_mb()
        if meminfo:
            ram_mb = meminfo
            notes.append(f"ram_mb_from_proc={ram_mb}")
        else:
            ram_mb = max(1024, cpu * 1024)
            notes.append("ram_mb_estimated_from_cpu")
    except Exception as exc:  # noqa: BLE001
        ram_mb = max(1024, cpu * 1024)
        notes.append(f"ram_detect_fallback:{exc!r}")

    gpu_count = 0
    gpu_names: list[str] = []
    vram_total = 0
    has_cuda = False
    try:
        from ocr_local.infra.gpu_optimization import GpuOptimizer  # noqa: WPS433

        opt = GpuOptimizer()
        caps = opt.detect_capabilities()
        if caps:
            has_cuda = True
            gpu_count = len(caps)
            gpu_names = [c.name for c in caps]
            vram_total = sum(int(c.memory_total_mb) for c in caps)
            notes.append(f"cuda_gpus={gpu_count} vram_total_mb={vram_total}")
        else:
            notes.append("no_cuda_gpus")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"gpu_detect_error:{exc!r}")

    return HostHardware(
        cpu_count=cpu,
        ram_mb=int(ram_mb),
        gpu_count=gpu_count,
        gpu_names=tuple(gpu_names),
        vram_total_mb=int(vram_total),
        has_cuda=has_cuda,
        detection_notes=tuple(notes),
    )


def _read_meminfo_mb() -> int | None:
    """Read MemTotal kB from /proc/meminfo → MB."""
    path = "/proc/meminfo"
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024)
    except Exception:  # noqa: BLE001
        return None
    return None


def recommend_pipeline_profile(hw: HostHardware | None = None) -> PipelineTuneProfile:
    """Recommend pipeline knobs from host hardware.

    Heuristics (conservative, quality-first):
    - GPU present: workers scale with VRAM; extractors with CPU; DPI 300
    - CPU-only: fewer workers, DPI 300, smaller queues
    - Caps prevent oversubscription on tiny hosts
    """
    hw = hw or detect_host_hardware()
    rationale: list[str] = []

    if hw.has_cuda and hw.vram_total_mb > 0:
        # ~1 worker per 2GB VRAM, min 1 max 24
        workers = max(1, min(24, hw.vram_total_mb // 2048))
        extractors = max(2, min(16, hw.cpu_count // 2 or 1))
        compressors = max(2, min(12, hw.cpu_count // 2 or 1))
        dpi = 300
        max_batch = (
            4 if hw.vram_total_mb >= 16000 else (2 if hw.vram_total_mb >= 8000 else 1)
        )
        profile = f"solo-gpu-{hw.gpu_count}x-{hw.vram_total_mb}mb"
        rationale.append(
            f"GPU path: workers={workers} from VRAM {hw.vram_total_mb}MB (~2GB each)"
        )
        rationale.append(f"extractors={extractors} from cpu_count={hw.cpu_count}")
    else:
        workers = max(1, min(8, max(1, hw.cpu_count // 2)))
        extractors = max(1, min(8, max(1, hw.cpu_count // 2)))
        compressors = max(1, min(6, max(1, hw.cpu_count // 3)))
        dpi = 300
        max_batch = 1
        profile = f"solo-cpu-{hw.cpu_count}c-{hw.ram_mb}mb"
        rationale.append(
            f"CPU path: workers={workers} extractors={extractors} "
            f"cpu={hw.cpu_count} ram_mb={hw.ram_mb}"
        )

    # Queue sizes: buffer ~ workers * 4 images, cap by RAM (~20MB/page estimate)
    pages_budget = max(8, min(200, (hw.ram_mb // 20) // 2))
    image_q = max(8, min(200, pages_budget))
    chunk_q = max(4, min(50, image_q // 4))
    rationale.append(f"queues image={image_q} chunk={chunk_q} from ram budget")

    return PipelineTuneProfile(
        num_extractors=extractors,
        num_workers=workers,
        num_compressors=compressors,
        image_queue_size=image_q,
        chunk_queue_size=chunk_q,
        dpi=dpi,
        max_batch_images=max_batch,
        profile_name=profile,
        rationale=tuple(rationale),
        hardware={
            "cpu_count": hw.cpu_count,
            "ram_mb": hw.ram_mb,
            "gpu_count": hw.gpu_count,
            "gpu_names": list(hw.gpu_names),
            "vram_total_mb": hw.vram_total_mb,
            "has_cuda": hw.has_cuda,
            "detection_notes": list(hw.detection_notes),
        },
    )


def autotune_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Return True when OCR_AUTO_TUNE is explicitly on (default False)."""
    e = env if env is not None else os.environ
    return str(e.get("OCR_AUTO_TUNE", "")).lower().strip() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Explicit env keys that suppress auto-tune for that knob (precedence rule).
_EXPLICIT_KNOB_ENV: dict[str, str] = {
    "num_extractors": "NUM_EXTRACTORS",
    "num_workers": "NUM_WORKERS",
    "num_compressors": "NUM_COMPRESSORS",
    "image_queue_size": "IMAGE_QUEUE_SIZE",
    "chunk_queue_size": "CHUNK_QUEUE_SIZE",
    "dpi": "DPI",
}


def profile_to_pipeline_env(profile: PipelineTuneProfile) -> dict[str, str]:
    """Map a tune profile to NUM_* / queue env vars (pipeline form)."""
    return {
        "NUM_EXTRACTORS": str(profile.num_extractors),
        "NUM_WORKERS": str(profile.num_workers),
        "NUM_COMPRESSORS": str(profile.num_compressors),
        "IMAGE_QUEUE_SIZE": str(profile.image_queue_size),
        "CHUNK_QUEUE_SIZE": str(profile.chunk_queue_size),
        "DPI": str(profile.dpi),
    }


def profile_to_solo_env(profile: PipelineTuneProfile) -> dict[str, str]:
    """Map a tune profile to SOLO_* compose knobs (docker-compose.solo.yml)."""
    return {
        "SOLO_NUM_EXTRACTORS": str(profile.num_extractors),
        "SOLO_NUM_WORKERS": str(profile.num_workers),
        "SOLO_NUM_COMPRESSORS": str(profile.num_compressors),
        "SOLO_IMAGE_QUEUE_SIZE": str(profile.image_queue_size),
        "SOLO_PROFILE_NAME": profile.profile_name,
    }


def solo_best_fit_profile(hw: HostHardware | None = None) -> PipelineTuneProfile:
    """ES-2: SOLO one-machine best-fit = ES-1 recommend on this host."""
    return recommend_pipeline_profile(hw)


def apply_profile_to_pipeline_config(
    cfg: Any,
    profile: PipelineTuneProfile,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return a copy of *cfg* with auto-tune knobs applied.

    Explicit env knobs (NUM_WORKERS, etc.) always win — auto-tune only fills
    knobs the operator did not set. Works with :class:`PipelineConfig`.
    """
    import dataclasses

    e = env if env is not None else os.environ
    updates: dict[str, Any] = {}
    candidates = {
        "num_extractors": profile.num_extractors,
        "num_workers": profile.num_workers,
        "num_compressors": profile.num_compressors,
        "image_queue_size": profile.image_queue_size,
        "chunk_queue_size": profile.chunk_queue_size,
        "dpi": profile.dpi,
    }
    for field_name, value in candidates.items():
        env_key = _EXPLICIT_KNOB_ENV[field_name]
        if env_key in e:
            continue  # explicit wins
        if hasattr(cfg, field_name):
            updates[field_name] = value
    if not updates:
        return cfg
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.replace(cfg, **updates)
    # Fallback for simple namespace objects
    for k, v in updates.items():
        setattr(cfg, k, v)
    return cfg


def maybe_apply_autotune_to_pipeline_config(
    cfg: Any,
    env: Mapping[str, str] | None = None,
) -> tuple[Any, PipelineTuneProfile | None]:
    """If OCR_AUTO_TUNE is on, apply SOLO best-fit knobs; else return cfg unchanged.

    Returns ``(config, profile_or_None)``.
    """
    e = env if env is not None else os.environ
    if not autotune_enabled_from_env(e):
        return cfg, None
    profile = solo_best_fit_profile()
    applied = apply_profile_to_pipeline_config(cfg, profile, env=e)
    logger.info(
        "OCR_AUTO_TUNE applied SOLO profile %s (workers=%s extractors=%s dpi=%s); "
        "explicit NUM_* env keys still take precedence",
        profile.profile_name,
        applied.num_workers if hasattr(applied, "num_workers") else "?",
        applied.num_extractors if hasattr(applied, "num_extractors") else "?",
        applied.dpi if hasattr(applied, "dpi") else "?",
    )
    return applied, profile
