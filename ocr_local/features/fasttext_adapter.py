"""Shared FastText loading and prediction helpers.

This module intentionally preserves the current ``fasttext.load_model`` /
``model.predict`` contract while isolating the optional C++ dependency behind
one import boundary.  That keeps the Python 3.12 migration path narrow: future
replacement packages can adapt here without touching OCR pipeline call sites.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from typing import Any, Protocol


class LoggerLike(Protocol):
    def warning(self, msg: str, *args: object) -> None: ...


def model_sha256(model_path: str) -> str:
    """Return the SHA-256 for ``model_path`` or ``""`` when unavailable."""
    if not model_path or not os.path.exists(model_path):
        return ""
    try:
        with open(model_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def load_model(
    model_path: str,
    *,
    logger: LoggerLike | None = None,
    context: str = "FastText",
    attach_sha256: bool = False,
) -> Any:
    """Load a FastText model, returning ``None`` for all optional-dep failures."""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        fasttext = importlib.import_module("fasttext")
    except Exception as exc:
        if logger is not None:
            logger.warning("%s import failed: %s", context, exc)
        return None
    try:
        model = fasttext.load_model(model_path)
    except Exception as exc:
        if logger is not None:
            logger.warning("%s load_model failed for %r: %s", context, model_path, exc)
        return None
    if attach_sha256:
        try:
            setattr(model, "_sha256", model_sha256(model_path))
        except Exception:
            pass
    return model


def predict_language(
    model: Any,
    text: str,
    *,
    k: int = 1,
    logger: LoggerLike | None = None,
    context: str = "FastText predict",
) -> tuple[str | None, float]:
    """Return ``(fasttext_code, score)`` from a FastText-style model.

    The helper accepts the existing ``model.predict(text, k=N)`` response shape:
    ``(["__label__en"], [0.99])``.  It deliberately returns the raw FastText
    language code so callers keep their existing PaddleOCR mapping policy.
    """
    if model is None or not text:
        return None, 0.0
    try:
        labels, scores = model.predict(text, k=k)
    except Exception as exc:
        if logger is not None:
            logger.warning("%s failed: %s", context, exc)
        return None, 0.0
    try:
        raw_label = labels[0]
    except (IndexError, TypeError):
        return None, 0.0
    raw_label = raw_label if isinstance(raw_label, str) else str(raw_label)
    lang_code = raw_label.replace("__label__", "").strip().lower()
    if not lang_code:
        return None, 0.0
    try:
        score = float(scores[0])
    except (IndexError, TypeError, ValueError):
        return None, 0.0
    if not math.isfinite(score):
        return None, 0.0
    return lang_code, max(0.0, min(1.0, score))
