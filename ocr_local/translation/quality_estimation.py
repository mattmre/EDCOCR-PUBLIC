"""COMETKiwi reference-free translation quality estimation -- Plan B Wave 3 (B15).

This module wraps the ``unbabel-comet`` library to score translation
``(source, target)`` pairs without needing a human reference.  The
intended use is to attach per-span quality scores onto a finalised
:class:`DocumentTranslation`, then enrich the sidecar JSON with
aggregate and per-span scores so downstream consumers (review queue,
SLA dashboards, exception routing) can act on low-quality output.

Design notes
------------
* **Air-gapped friendly.**  The estimator never reaches out to the
  HuggingFace hub at runtime when ``model_path`` is set.  When
  ``model_path`` is missing and ``allow_download=False``, the estimator
  raises :class:`ModelNotCachedError` instead of silently downloading.
* **Lazy + guarded import.**  ``comet`` is optional.  When not
  installed, every public method returns a sentinel
  :class:`QualityScore` with ``available=False`` and never raises.
* **Self-contained config.**  ``QualityEstimationConfig`` is parsed from
  environment variables via :meth:`QualityEstimationConfig.from_env`;
  this module deliberately does NOT modify ``pipeline_config.py``
  (B17 owns that wiring in Wave 3).
* **Custody emission lives in a sibling module.**  See
  ``ocr_local.translation.quality_custody`` for the helper that emits
  quality custody events; this module is purely the inference + scoring
  surface so it can be imported without pulling custody at import time.

The threshold semantics map onto the existing
``ReasonCode.QUALITY_BELOW_THRESHOLD`` event used by the reviewer
pipeline:

    score < score_threshold_reject  -> hard reject (block downstream use)
    score < score_threshold_warn    -> soft warning (surface in dashboard)
    score >= score_threshold_warn   -> ok

The two thresholds are configurable per-deployment so the operator can
tune the false-positive / false-negative trade-off without code changes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ocr_local.translation.cache import ModelNotCachedError

if TYPE_CHECKING:
    from ocr_local.translation.models import DocumentTranslation

logger = logging.getLogger(__name__)

__all__ = [
    "CometKiwiEstimator",
    "DocumentQualityReport",
    "ModelNotCachedError",
    "QualityEstimationConfig",
    "QualityScore",
    "assess_document_quality",
    "enrich_translation_output",
]


# ---------------------------------------------------------------------------
# Lazy comet import
# ---------------------------------------------------------------------------


def _try_import_comet():
    """Attempt to import ``comet`` lazily.

    Returns the module on success, ``None`` on ``ImportError``.  Kept as a
    helper so tests can monkeypatch this function to stub the library
    without having ``unbabel-comet`` installed.
    """
    try:
        import comet  # type: ignore[import-not-found]
    except ImportError:
        return None
    return comet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclasses.dataclass(frozen=True)
class QualityEstimationConfig:
    """Configuration for the COMETKiwi estimator.

    Defaults match the upstream Unbabel WMT22 CometKiwi DA model and the
    operator-recommended thresholds (``warn=0.4``, ``reject=0.2``).
    """

    enabled: bool = False
    model_id: str = "Unbabel/wmt22-cometkiwi-da"
    model_path: str | None = None
    batch_size: int = 16
    device: str = "cpu"
    score_threshold_warn: float = 0.4
    score_threshold_reject: float = 0.2

    @classmethod
    def from_env(cls) -> "QualityEstimationConfig":
        """Build a config from ``OCR_TRANSLATION_QE_*`` environment vars."""
        return cls(
            enabled=_env_bool("OCR_TRANSLATION_QE_ENABLED", False),
            model_id=os.environ.get(
                "OCR_TRANSLATION_QE_MODEL_ID",
                "Unbabel/wmt22-cometkiwi-da",
            ),
            model_path=os.environ.get("OCR_TRANSLATION_QE_MODEL_PATH") or None,
            batch_size=_env_int("OCR_TRANSLATION_QE_BATCH_SIZE", 16),
            device=os.environ.get("OCR_TRANSLATION_QE_DEVICE", "cpu"),
            score_threshold_warn=_env_float(
                "OCR_TRANSLATION_QE_SCORE_THRESHOLD_WARN", 0.4
            ),
            score_threshold_reject=_env_float(
                "OCR_TRANSLATION_QE_SCORE_THRESHOLD_REJECT", 0.2
            ),
        )


# ---------------------------------------------------------------------------
# Score / report dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QualityScore:
    """Single ``(source, target)`` quality estimation result.

    ``score`` is the raw COMETKiwi output, normalised to ``[0, 1]``.
    When ``available=False`` (e.g. ``comet`` not installed, or the
    estimator failed to load), ``score`` is ``None`` and ``reason``
    explains why.
    """

    score: float | None
    available: bool
    reason: str | None = None
    model_id: str | None = None


@dataclasses.dataclass
class DocumentQualityReport:
    """Aggregate per-document quality breakdown."""

    model_id: str
    score_mean: float | None
    score_min: float | None
    threshold_warn: float
    threshold_reject: float
    threshold_warn_count: int
    threshold_reject_count: int
    per_span_scores: list[dict]
    page_means: dict[int, float | None]
    span_count: int
    scored_count: int

    def to_dict(self) -> dict:
        """Serialise for the ``quality_estimation`` JSON enrichment."""
        return {
            "model_id": self.model_id,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "threshold_warn": self.threshold_warn,
            "threshold_reject": self.threshold_reject,
            "threshold_warn_count": self.threshold_warn_count,
            "threshold_reject_count": self.threshold_reject_count,
            "span_count": self.span_count,
            "scored_count": self.scored_count,
            "page_means": {str(k): v for k, v in self.page_means.items()},
            "per_span_scores": list(self.per_span_scores),
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class CometKiwiEstimator:
    """Wrapper around the COMETKiwi inference model.

    The estimator loads the underlying model lazily on first call, so
    constructing the object is cheap and never raises when ``comet`` is
    absent (the failure mode is reflected in returned
    :class:`QualityScore` instances, never an exception).

    Parameters
    ----------
    config:
        Frozen :class:`QualityEstimationConfig`.
    allow_download:
        When False (the default, and the air-gapped requirement), the
        estimator refuses to fetch weights from the HuggingFace hub.
        ``model_path`` MUST be set in that mode -- otherwise the
        constructor raises :class:`ModelNotCachedError`.
    """

    def __init__(
        self,
        config: QualityEstimationConfig,
        *,
        allow_download: bool = False,
    ) -> None:
        self.config = config
        self.allow_download = allow_download
        self._model = None
        self._load_failed = False
        self._load_failed_reason: str | None = None

        comet = _try_import_comet()
        if comet is None:
            self._load_failed = True
            self._load_failed_reason = "comet_not_installed"
            logger.info(
                "CometKiwiEstimator: comet not installed -- estimator "
                "will return available=False for all calls."
            )
            return

        # Air-gapped guard: if no on-disk path AND we can't download,
        # raise eagerly so the caller gets a clear failure rather than
        # discovering it on first score.
        if not config.model_path and not allow_download:
            raise ModelNotCachedError(
                engine="cometkiwi",
                model_id=config.model_id,
                reason="model_path unset and allow_download=False",
            )

        # If ``model_path`` is set we expect the directory to exist.
        if config.model_path:
            mp = Path(config.model_path)
            if not mp.exists():
                raise ModelNotCachedError(
                    engine="cometkiwi",
                    model_id=config.model_id,
                    reason=f"model_path does not exist: {mp}",
                )

    # -- model loading ---------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load the underlying model on first use.

        Returns True when the model is ready to score, False otherwise.
        """
        if self._load_failed:
            return False
        if self._model is not None:
            return True

        comet = _try_import_comet()
        if comet is None:
            self._load_failed = True
            self._load_failed_reason = "comet_not_installed"
            return False

        try:
            if self.config.model_path:
                # Local checkpoint path -- no network required.  Standard COMET
                # bundles are loaded from a checkpoint file; standalone bundles
                # such as solailabs/wmt22-cometkiwi-da-int8 ship a load.py
                # helper plus state_dict.pt and are loaded via that helper.
                standalone = _load_standalone_cometkiwi(self.config.model_path)
                if standalone is not None:
                    self._model = standalone
                else:
                    load_from_checkpoint = getattr(
                        comet, "load_from_checkpoint", None
                    )
                    if load_from_checkpoint is None:  # pragma: no cover -- defensive
                        self._load_failed = True
                        self._load_failed_reason = "load_from_checkpoint missing"
                        return False
                    self._model = load_from_checkpoint(self.config.model_path)
            else:
                # download_model + load_from_checkpoint -- only reachable
                # when allow_download=True (constructor guards otherwise).
                download_model = getattr(comet, "download_model", None)
                load_from_checkpoint = getattr(
                    comet, "load_from_checkpoint", None
                )
                if download_model is None or load_from_checkpoint is None:
                    self._load_failed = True
                    self._load_failed_reason = "comet api missing"
                    return False
                ckpt = download_model(self.config.model_id)
                self._model = load_from_checkpoint(ckpt)
        except Exception as exc:  # pragma: no cover -- defensive
            self._load_failed = True
            self._load_failed_reason = f"load_failed: {exc}"
            logger.warning(
                "CometKiwiEstimator: failed to load model %s: %s",
                self.config.model_id, exc,
            )
            return False
        return True

    # -- scoring ---------------------------------------------------------

    def score_pair(self, source: str, target: str) -> QualityScore:
        """Score a single ``(source, target)`` pair.

        Always returns a :class:`QualityScore`; never raises.  Empty
        inputs short-circuit to a ``score=None, available=False`` result
        because COMETKiwi is undefined on empty strings.
        """
        if not source or not target:
            return QualityScore(
                score=None,
                available=False,
                reason="empty_input",
                model_id=self.config.model_id,
            )

        if not self._ensure_loaded():
            return QualityScore(
                score=None,
                available=False,
                reason=self._load_failed_reason or "estimator_unavailable",
                model_id=self.config.model_id,
            )

        try:
            results = self._predict([{"src": source, "mt": target}])
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "CometKiwiEstimator.score_pair failed: %s", exc,
            )
            return QualityScore(
                score=None,
                available=False,
                reason=f"predict_failed: {exc}",
                model_id=self.config.model_id,
            )

        if not results:
            return QualityScore(
                score=None,
                available=False,
                reason="empty_predict_result",
                model_id=self.config.model_id,
            )
        return QualityScore(
            score=_clamp_score(results[0]),
            available=True,
            reason=None,
            model_id=self.config.model_id,
        )

    def score_batch(
        self, pairs: list[tuple[str, str]],
    ) -> list[QualityScore]:
        """Score a batch of ``(source, target)`` pairs.

        Pairs with empty inputs receive ``available=False`` results
        without involving the model.  When the model is not available,
        every pair receives the same load-failure result.
        """
        if not pairs:
            return []

        if not self._ensure_loaded():
            reason = self._load_failed_reason or "estimator_unavailable"
            return [
                QualityScore(
                    score=None,
                    available=False,
                    reason=reason,
                    model_id=self.config.model_id,
                )
                for _ in pairs
            ]

        # Pre-filter empty inputs and remember their indices so we can
        # splice the results back together in order.
        nonempty_inputs: list[dict] = []
        index_map: list[int | None] = []
        out: list[QualityScore | None] = [None] * len(pairs)

        for i, (src, tgt) in enumerate(pairs):
            if not src or not tgt:
                out[i] = QualityScore(
                    score=None,
                    available=False,
                    reason="empty_input",
                    model_id=self.config.model_id,
                )
                index_map.append(None)
            else:
                index_map.append(len(nonempty_inputs))
                nonempty_inputs.append({"src": src, "mt": tgt})

        if nonempty_inputs:
            try:
                raw_scores = self._predict(nonempty_inputs)
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning("CometKiwiEstimator.score_batch failed: %s", exc)
                fallback = QualityScore(
                    score=None,
                    available=False,
                    reason=f"predict_failed: {exc}",
                    model_id=self.config.model_id,
                )
                for idx in range(len(out)):
                    if out[idx] is None:
                        out[idx] = fallback
                return [s for s in out if s is not None]
        else:
            raw_scores = []

        for idx, mapped in enumerate(index_map):
            if mapped is None:
                continue
            if mapped >= len(raw_scores):
                out[idx] = QualityScore(
                    score=None,
                    available=False,
                    reason="missing_score",
                    model_id=self.config.model_id,
                )
            else:
                out[idx] = QualityScore(
                    score=_clamp_score(raw_scores[mapped]),
                    available=True,
                    reason=None,
                    model_id=self.config.model_id,
                )
        return [s for s in out if s is not None]

    # -- prediction shim -------------------------------------------------

    def _predict(self, payload: list[dict]) -> list[float]:
        """Call the model's predict API and return raw scores.

        Hidden behind a helper so tests can monkeypatch a single method
        on the estimator instance instead of mocking the entire comet
        library.  Real comet returns an object with a ``.scores`` list;
        we tolerate either shape.
        """
        if self._model is None:  # pragma: no cover -- guarded by callers
            return []
        result = self._model.predict(
            payload,
            batch_size=self.config.batch_size,
            gpus=1 if self.config.device == "cuda" else 0,
        )
        # comet returns a Prediction dataclass with ``.scores``; tests may
        # return a plain list.  Normalise to ``list[float]``.
        if hasattr(result, "scores"):
            return list(result.scores)
        if isinstance(result, dict) and "scores" in result:
            return list(result["scores"])
        if isinstance(result, list):
            return list(result)
        return []


def _clamp_score(value: object) -> float | None:
    """Clamp a raw model output to ``[0, 1]``.

    Returns ``None`` for non-numeric inputs (defensive against odd
    shapes from monkeypatched fakes).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _load_standalone_cometkiwi(model_path: str) -> object | None:
    """Load a standalone COMETKiwi directory when one is supplied.

    Some local/dev evidence bundles are not single ``.ckpt`` files.  The
    ungated int8 COMETKiwi bundle used by the deployed E2E lane contains
    ``load.py`` with a ``load_model`` entry point and local weights.  This
    helper keeps that path explicit while preserving the standard COMET
    checkpoint path for production-approved bundles.
    """
    path = Path(model_path)
    loader = path / "load.py"
    state_dict = path / "state_dict.pt"
    if not path.is_dir() or not loader.exists() or not state_dict.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "_ocr_local_standalone_cometkiwi_loader",
        loader,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import standalone COMETKiwi loader: {loader}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_model = getattr(module, "load_model", None)
    if load_model is None:
        raise RuntimeError(f"standalone COMETKiwi loader missing load_model: {loader}")
    return load_model(path)


# ---------------------------------------------------------------------------
# Document aggregation
# ---------------------------------------------------------------------------


def assess_document_quality(
    doc: "DocumentTranslation",
    estimator: CometKiwiEstimator,
) -> DocumentQualityReport:
    """Score every span in ``doc`` and aggregate into a report.

    Spans for which the estimator returns ``available=False`` (e.g. empty
    text, model not loaded) are recorded with ``score=None`` in
    ``per_span_scores`` and excluded from mean/min calculations.
    """
    pairs: list[tuple[str, str]] = []
    span_pointers: list[tuple[int, int, str]] = []
    # ``span_pointers`` maps each batched pair back to (page_num, span_index, span_id).

    for page in doc.pages:
        for span_idx, span in enumerate(page.spans):
            src = getattr(span, "source_text", "")
            tgt = getattr(span, "target_text", "")
            span_id = getattr(span, "span_id", f"p{page.page_num}_s{span_idx}")
            pairs.append((src or "", tgt or ""))
            span_pointers.append((page.page_num, span_idx, span_id))

    if pairs:
        scores = estimator.score_batch(pairs)
    else:
        scores = []

    per_span: list[dict] = []
    page_buckets: dict[int, list[float]] = {}
    all_scores: list[float] = []
    warn_count = 0
    reject_count = 0
    scored_count = 0

    threshold_warn = estimator.config.score_threshold_warn
    threshold_reject = estimator.config.score_threshold_reject

    for (page_num, span_idx, span_id), score in zip(span_pointers, scores):
        per_span.append(
            {
                "page_num": page_num,
                "span_index": span_idx,
                "span_id": span_id,
                "score": score.score,
                "available": score.available,
                "reason": score.reason,
            }
        )
        if score.available and score.score is not None:
            all_scores.append(score.score)
            page_buckets.setdefault(page_num, []).append(score.score)
            scored_count += 1
            if score.score < threshold_reject:
                reject_count += 1
            elif score.score < threshold_warn:
                warn_count += 1

    score_mean = (
        sum(all_scores) / len(all_scores) if all_scores else None
    )
    score_min = min(all_scores) if all_scores else None
    page_means = {
        page: (sum(buckets) / len(buckets) if buckets else None)
        for page, buckets in page_buckets.items()
    }

    return DocumentQualityReport(
        model_id=estimator.config.model_id,
        score_mean=score_mean,
        score_min=score_min,
        threshold_warn=threshold_warn,
        threshold_reject=threshold_reject,
        threshold_warn_count=warn_count,
        threshold_reject_count=reject_count,
        per_span_scores=per_span,
        page_means=page_means,
        span_count=len(per_span),
        scored_count=scored_count,
    )


# ---------------------------------------------------------------------------
# JSON enrichment
# ---------------------------------------------------------------------------


def enrich_translation_output(
    translation_json: dict,
    report: DocumentQualityReport,
) -> dict:
    """Attach ``quality_estimation`` to the translation JSON in place.

    The function is **idempotent**: calling it twice with the same
    report leaves the same result.  The existing ``quality`` block
    (engine-declared aggregate) is left untouched -- the QE output is
    always written under the dedicated ``quality_estimation`` key so
    consumers can distinguish the engine self-report from the
    independent QE assessment.
    """
    translation_json["quality_estimation"] = report.to_dict()
    return translation_json
