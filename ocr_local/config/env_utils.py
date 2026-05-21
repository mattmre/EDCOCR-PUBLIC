"""Centralized environment variable schema and typed accessors.

This module provides:
- Typed accessor helpers (get_env, get_env_int, get_env_float, get_env_bool)
- A registry of all known env vars with their types, defaults, and descriptions
- Startup validation (log warnings for required vars that are unset)

Usage:
    from env_utils import get_env_int, get_env_bool
    NUM_WORKERS = get_env_int("NUM_WORKERS", 12)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def get_env(name: str, default: str = "") -> str:
    """Get a string env var with a default."""
    return os.environ.get(name, default)


def get_env_int(name: str, default: int) -> int:
    """Get an integer env var; logs a warning on parse failure."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(
            "Env var %s=%r is not a valid int; using default %d", name, val, default
        )
        return default


def get_env_float(name: str, default: float) -> float:
    """Get a float env var; logs a warning on parse failure."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(
            "Env var %s=%r is not a valid float; using default %f",
            name,
            val,
            default,
        )
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get a boolean env var. Truthy: '1', 'true', 'yes' (case-insensitive)."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Known env var registry -- documents all env vars used in the codebase
# Each entry: (type, default, description, required)
# ---------------------------------------------------------------------------

ENV_SCHEMA: dict[str, dict[str, Any]] = {
    # Pipeline core
    "OCR_OUTPUT_DIR": {
        "type": str,
        "default": "ocr_output",
        "description": "Output directory for OCR results",
    },
    "OCR_SOURCE_DIR": {
        "type": str,
        "default": "ocr_source",
        "description": "Input directory for source documents",
    },
    "NUM_WORKERS": {
        "type": int,
        "default": 12,
        "description": "Number of GPU OCR worker threads",
    },
    "NUM_EXTRACTORS": {
        "type": int,
        "default": 8,
        "description": "Number of CPU extractor threads",
    },
    "NUM_COMPRESSORS": {
        "type": int,
        "default": 8,
        "description": "Number of Ghostscript compression threads",
    },
    "DPI": {
        "type": int,
        "default": 300,
        "description": "Base DPI for OCR rendering",
    },
    "IMAGE_QUEUE_SIZE": {
        "type": int,
        "default": 200,
        "description": "Max images buffered between extractor and worker",
    },
    "CHUNK_QUEUE_SIZE": {
        "type": int,
        "default": 50,
        "description": "Max chunks buffered from scheduler to extractor",
    },
    # Language / model
    "OCR_LANGUAGE": {
        "type": str,
        "default": "en",
        "description": "Default OCR language",
    },
    "OCR_LANGUAGE_TIERS": {
        "type": str,
        "default": "core",
        "description": "Language tiers to load: core, extended",
    },
    "FASTTEXT_MODEL_PATH": {
        "type": str,
        "default": "",
        "description": "Path to FastText lid.176.bin model",
    },
    # Feature flags
    "ENABLE_DOCINTEL": {
        "type": bool,
        "default": False,
        "description": "Enable PP-StructureV3 document intelligence",
    },
    "ENABLE_NER": {
        "type": bool,
        "default": False,
        "description": "Enable spaCy named entity recognition",
    },
    "ENABLE_HANDWRITING": {
        "type": bool,
        "default": False,
        "description": "Enable handwriting detection",
    },
    "ENABLE_CLASSIFICATION": {
        "type": bool,
        "default": False,
        "description": "Enable document classification",
    },
    "ENABLE_EXTRACTION": {
        "type": bool,
        "default": False,
        "description": "Enable structured field extraction",
    },
    "ENABLE_VERTICAL_TEXT": {
        "type": bool,
        "default": False,
        "description": "Enable CJK vertical text detection",
    },
    "ENABLE_SIGNATURE_VERIFICATION": {
        "type": bool,
        "default": False,
        "description": "Enable experimental signature verification",
    },
    "ENABLE_RETRIEVAL_OUTPUT": {
        "type": bool,
        "default": False,
        "description": "Enable unified retrieval JSON+Markdown output",
    },
    "ENABLE_EXCEPTION_ROUTING": {
        "type": bool,
        "default": False,
        "description": "Enable confidence-based exception routing",
    },
    "ENABLE_GPU_OPTIMIZATION": {
        "type": bool,
        "default": False,
        "description": "Enable GPU kernel fusion optimization",
    },
    "ENABLE_PAGE_ROUTING": {
        "type": bool,
        "default": False,
        "description": "Enable smart page-to-backend routing",
    },
    "ENABLE_NOISE_PROFILING": {
        "type": bool,
        "default": False,
        "description": "Enable adaptive noise profiling preprocessing",
    },
    # API
    "OCR_API_KEY": {
        "type": str,
        "default": "",
        "description": "API authentication key (required for auth mode)",
        "required": True,
    },
    "API_HOST": {
        "type": str,
        "default": "0.0.0.0",
        "description": "API server host",
    },
    "API_PORT": {
        "type": int,
        "default": 8000,
        "description": "API server port",
    },
    "MAX_CONCURRENT_JOBS": {
        "type": int,
        "default": 10,
        "description": "Maximum concurrent OCR jobs",
    },
    "ALLOW_UNAUTHENTICATED": {
        "type": bool,
        "default": False,
        "description": "Allow unauthenticated API access",
    },
    # SSE / WebSocket
    "SSE_POLL_INTERVAL": {
        "type": float,
        "default": 30.0,
        "description": "SSE fallback poll interval in seconds",
    },
    "SSE_STREAM_TIMEOUT": {
        "type": float,
        "default": 1800.0,
        "description": "SSE stream maximum duration in seconds",
    },
    # Storage / S3
    "S3_ENDPOINT": {
        "type": str,
        "default": "",
        "description": "S3-compatible endpoint URL",
    },
    "S3_BUCKET": {
        "type": str,
        "default": "",
        "description": "S3 bucket name for job storage",
    },
    "S3_ACCESS_KEY": {
        "type": str,
        "default": "",
        "description": "S3 access key ID",
    },
    "S3_SECRET_KEY": {
        "type": str,
        "default": "",
        "description": "S3 secret access key",
    },
    # Celery / Redis
    "CELERY_BROKER_URL": {
        "type": str,
        "default": "",
        "description": "Celery broker URL (RabbitMQ AMQP)",
        "required": True,
    },
    "REDIS_URL": {
        "type": str,
        "default": "",
        "description": "Redis URL for Celery results backend",
    },
    # Monitoring
    "METRICS_API_KEY": {
        "type": str,
        "default": "",
        "description": "API key for /api/v1/metrics/ and /api/v1/prometheus/",
    },
    "DEPLOYMENT_ENV": {
        "type": str,
        "default": "development",
        "description": "Deployment environment (development/staging/production)",
    },
    # OCR engine routing
    "OCR_TASK_ROUTING": {
        "type": str,
        "default": "gpu",
        "description": "OCR task queue routing: gpu, cpu, auto",
    },
    "OCR_ENGINE_SELECTION": {
        "type": str,
        "default": "paddle",
        "description": "Engine selection per-page: auto, paddle, tesseract",
    },
}


def validate_env(required_only: bool = False) -> list[str]:
    """Return list of warning messages for misconfigured env vars.

    Args:
        required_only: If True, only check vars marked required=True.

    Returns:
        List of warning strings (empty = all OK).
    """
    warnings: list[str] = []
    for name, meta in ENV_SCHEMA.items():
        val = os.environ.get(name)
        if meta.get("required") and not val:
            warnings.append(f"Required env var {name} is not set")
        elif not required_only and val is not None:
            # Type-check set values
            typ = meta["type"]
            if typ is int:
                try:
                    int(val)
                except ValueError:
                    warnings.append(
                        f"Env var {name}={val!r} cannot be parsed as int"
                    )
            elif typ is float:
                try:
                    float(val)
                except ValueError:
                    warnings.append(
                        f"Env var {name}={val!r} cannot be parsed as float"
                    )
            elif typ is bool:
                if val.strip().lower() not in (
                    "0",
                    "1",
                    "true",
                    "false",
                    "yes",
                    "no",
                    "",
                ):
                    warnings.append(
                        f"Env var {name}={val!r} is not a recognized bool value"
                    )
    return warnings
