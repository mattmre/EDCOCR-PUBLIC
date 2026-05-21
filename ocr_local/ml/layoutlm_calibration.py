"""Confidence calibration for LayoutLMv3 semantic field extraction.

Softmax outputs from token-classification heads are often over-confident.
This module provides post-hoc calibration transforms (temperature scaling,
Platt scaling, isotonic regression) that map raw model probabilities to
well-calibrated confidence scores suitable for ensemble weighting in
``layoutlm_structure.py`` and downstream decision-making.

CTC-safe: this module operates on **pre-computed probabilities** from
token-classification (BIO tagging) models only — no text generation, no
causal LM heads, no torch imports.

Design goals:
- Pure Python core — ``numpy`` and ``sklearn`` are **lazy** imports.
- Temperature scaling works without any external dependency.
- Calibration parameters are JSON-serialisable for persistence.
- Environment-driven defaults via ``LAYOUTLM_CALIBRATION_METHOD`` and
  ``LAYOUTLM_CALIBRATION_PATH``.

Typical usage::

    from layoutlm_calibration import (
        CalibrationConfig,
        CalibrationMethod,
        ConfidenceCalibrator,
        calibrate_entity_confidence,
    )

    config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE_SCALING,
                               temperature=1.5)
    calibrator = ConfidenceCalibrator(config)
    calibrated_entities = calibrate_entity_confidence(raw_entities, calibrator)

Environment Variables:
    LAYOUTLM_CALIBRATION_METHOD (str):
        Calibration method to use.  One of ``"none"``, ``"temperature"``,
        ``"platt"``, ``"isotonic"``.  Default: ``"none"``.
    LAYOUTLM_CALIBRATION_PATH (str):
        Path to a saved calibration parameters JSON file.  When set, the
        default calibrator will load parameters from this file.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

LAYOUTLM_CALIBRATION_METHOD: str = os.environ.get(
    "LAYOUTLM_CALIBRATION_METHOD", "none"
).lower()

LAYOUTLM_CALIBRATION_PATH: str = os.environ.get(
    "LAYOUTLM_CALIBRATION_PATH", ""
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class CalibrationMethod(enum.Enum):
    """Supported confidence calibration methods."""

    NONE = "none"
    TEMPERATURE_SCALING = "temperature"
    PLATT_SCALING = "platt"
    ISOTONIC = "isotonic"


# Map env-var string values to enum members
_METHOD_LOOKUP: Dict[str, CalibrationMethod] = {
    m.value: m for m in CalibrationMethod
}


def _resolve_method(name: str) -> CalibrationMethod:
    """Resolve a method name string to a CalibrationMethod enum member."""
    return _METHOD_LOOKUP.get(name.lower(), CalibrationMethod.NONE)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration.

    Attributes:
        method:              Which calibration transform to apply.
        temperature:         Temperature scalar for temperature scaling
                             (T > 1 softens, T < 1 sharpens).  Default 1.0
                             (identity).
        platt_a:             Slope parameter *a* for Platt scaling.
        platt_b:             Intercept parameter *b* for Platt scaling.
        calibration_data_path:
            Optional path to a validation-set JSON file used by
            :meth:`ConfidenceCalibrator.fit`.
    """

    method: CalibrationMethod = CalibrationMethod.NONE
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    calibration_data_path: str = ""

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        d = asdict(self)
        d["method"] = self.method.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationConfig":
        """Reconstruct from a plain dict (e.g. loaded from JSON)."""
        d = dict(d)  # shallow copy
        d["method"] = _resolve_method(str(d.get("method", "none")))
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Pure-Python math helpers
# ---------------------------------------------------------------------------


def _softmax(values: List[float]) -> List[float]:
    """Numerically-stable softmax over a list of floats (pure Python)."""
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total == 0.0:
        n = len(values)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid (pure Python)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# ConfidenceCalibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Apply post-hoc confidence calibration to model outputs.

    Supports temperature scaling, Platt scaling, and isotonic regression.
    All methods operate on pre-computed logits or probabilities — no
    ``torch`` dependency required.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        if config is None:
            config = CalibrationConfig()
        self.config = config
        # Isotonic regression model (lazy-loaded via fit or load)
        self._isotonic_model: Any = None

    # ------------------------------------------------------------------
    # Core calibration
    # ------------------------------------------------------------------

    def calibrate(self, logits_or_probs: List[float]) -> List[float]:
        """Apply the configured calibration transform.

        Args:
            logits_or_probs: A list of raw logits **or** softmax
                probabilities, depending on the method:

                - ``TEMPERATURE_SCALING``: expects raw logits (pre-softmax).
                  Returns ``softmax(logits / T)``.
                - ``PLATT_SCALING``: expects raw logits.  Returns per-element
                  sigmoid-transformed values.
                - ``ISOTONIC``: expects probabilities.  Returns isotonic-
                  regression-transformed values.
                - ``NONE``: returns input unchanged.

        Returns:
            Calibrated probability list of the same length.
        """
        if not logits_or_probs:
            return []

        method = self.config.method

        if method == CalibrationMethod.NONE:
            return list(logits_or_probs)

        if method == CalibrationMethod.TEMPERATURE_SCALING:
            return self._temperature_scale(logits_or_probs)

        if method == CalibrationMethod.PLATT_SCALING:
            return self._platt_scale(logits_or_probs)

        if method == CalibrationMethod.ISOTONIC:
            return self._isotonic_scale(logits_or_probs)

        # Fallback — unknown method → identity
        logger.warning("Unknown calibration method %s; returning raw values.",
                       method)
        return list(logits_or_probs)

    # ------------------------------------------------------------------
    # Transform implementations
    # ------------------------------------------------------------------

    def _temperature_scale(self, logits: List[float]) -> List[float]:
        """Temperature scaling: softmax(logits / T)."""
        t = self.config.temperature
        if t <= 0:
            logger.warning("Temperature must be > 0 (got %.4f); using 1.0.", t)
            t = 1.0
        scaled = [v / t for v in logits]
        return _softmax(scaled)

    def _platt_scale(self, logits: List[float]) -> List[float]:
        """Platt scaling: σ(a * logit + b) per element."""
        a = self.config.platt_a
        b = self.config.platt_b
        return [_sigmoid(a * v + b) for v in logits]

    def _isotonic_scale(self, probs: List[float]) -> List[float]:
        """Isotonic regression transform (requires sklearn)."""
        if self._isotonic_model is None:
            logger.warning(
                "Isotonic model not fitted; returning raw probabilities."
            )
            return list(probs)
        try:
            import numpy as np  # lazy
            arr = np.array(probs, dtype=np.float64)
            calibrated = self._isotonic_model.predict(arr)
            return calibrated.tolist()
        except ImportError:
            logger.warning(
                "numpy not available for isotonic prediction; "
                "returning raw probabilities."
            )
            return list(probs)
        except Exception as exc:
            logger.warning("Isotonic prediction failed: %s", exc)
            return list(probs)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        validation_predictions: List[Dict[str, Any]],
        validation_labels: List[str],
    ) -> None:
        """Fit calibration parameters from validation data.

        Each entry in *validation_predictions* is a dict with at least
        a ``"confidence"`` key (float, the raw model probability) and
        optionally ``"logit"`` (float, the raw logit).

        *validation_labels* contains the ground-truth label strings
        (same length as *validation_predictions*).  For binary
        calibration the label is compared with the predicted label to
        produce a 0/1 correctness target.

        This method updates ``self.config`` in-place with fitted params.

        Args:
            validation_predictions: List of prediction dicts.
            validation_labels:      Corresponding ground-truth labels.
        """
        if not validation_predictions or not validation_labels:
            logger.warning("Empty validation data; skipping calibration fit.")
            return

        n = min(len(validation_predictions), len(validation_labels))
        predictions = validation_predictions[:n]
        labels = validation_labels[:n]

        # Build parallel arrays -------------------------------------------------
        confidences: List[float] = []
        logits: List[float] = []
        targets: List[int] = []

        for pred, true_label in zip(predictions, labels):
            conf = float(pred.get("confidence", 0.0))
            logit = float(pred.get("logit", conf))  # fallback to conf
            pred_label = pred.get("label", "")
            correct = 1 if pred_label == true_label else 0
            confidences.append(conf)
            logits.append(logit)
            targets.append(correct)

        method = self.config.method

        if method == CalibrationMethod.TEMPERATURE_SCALING:
            self._fit_temperature(logits, targets)
        elif method == CalibrationMethod.PLATT_SCALING:
            self._fit_platt(logits, targets)
        elif method == CalibrationMethod.ISOTONIC:
            self._fit_isotonic(confidences, targets)
        else:
            logger.info("Calibration method is NONE; nothing to fit.")

    def _fit_temperature(
        self, logits: List[float], targets: List[int]
    ) -> None:
        """Learn optimal temperature T by grid search on NLL.

        Uses a simple grid search (pure Python) over candidate T values
        to minimise negative log-likelihood on the validation set.
        """
        best_t = 1.0
        best_nll = float("inf")
        # Coarse search then fine search
        for t_candidate in [
            v / 10.0
            for v in range(1, 101)  # 0.1 .. 10.0 step 0.1
        ]:
            nll = 0.0
            for logit, target in zip(logits, targets):
                p = _sigmoid(logit / t_candidate)
                p_target = p if target == 1 else (1.0 - p)
                p_target = max(p_target, 1e-15)
                nll -= math.log(p_target)
            if nll < best_nll:
                best_nll = nll
                best_t = t_candidate

        self.config.temperature = best_t
        logger.info("Temperature scaling fit: T=%.4f (NLL=%.4f)", best_t,
                    best_nll)

    def _fit_platt(self, logits: List[float], targets: List[int]) -> None:
        """Fit Platt scaling parameters (a, b) via gradient descent.

        Uses a basic gradient descent loop (pure Python) to learn
        a and b that minimise binary cross-entropy on the validation
        set.
        """
        a = 1.0
        b = 0.0
        lr = 0.01
        for _ in range(1000):
            grad_a = 0.0
            grad_b = 0.0
            for logit, target in zip(logits, targets):
                p = _sigmoid(a * logit + b)
                err = p - target
                grad_a += err * logit
                grad_b += err
            n = len(logits)
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.config.platt_a = a
        self.config.platt_b = b
        logger.info("Platt scaling fit: a=%.6f, b=%.6f", a, b)

    def _fit_isotonic(
        self, confidences: List[float], targets: List[int]
    ) -> None:
        """Fit isotonic regression model via sklearn (lazy import)."""
        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression  # lazy
        except ImportError:
            logger.warning(
                "sklearn/numpy not available; cannot fit isotonic model."
            )
            return

        X = np.array(confidences, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X, y)
        self._isotonic_model = ir
        logger.info("Isotonic regression fit on %d samples.", len(confidences))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save calibration parameters to a JSON file.

        For isotonic regression the fitted model's internal arrays are
        serialised alongside the config dict.
        """
        data: Dict[str, Any] = {"config": self.config.to_dict()}

        # Persist isotonic model state (numpy arrays → lists)
        if self._isotonic_model is not None:
            try:
                iso_state: Dict[str, Any] = {}
                for attr in ("X_thresholds_", "y_thresholds_",
                             "X_min_", "X_max_"):
                    val = getattr(self._isotonic_model, attr, None)
                    if val is not None:
                        import numpy as np  # lazy
                        if isinstance(val, np.ndarray):
                            iso_state[attr] = val.tolist()
                        else:
                            iso_state[attr] = float(val)
                data["isotonic_state"] = iso_state
            except Exception as exc:
                logger.warning("Could not serialise isotonic state: %s", exc)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Calibration parameters saved to %s", path)

    def load(self, path: str) -> None:
        """Load calibration parameters from a JSON file.

        Restores config values and, if present, the isotonic regression
        model state.
        """
        p = Path(path)
        if not p.is_file():
            logger.warning("Calibration file not found: %s", path)
            return

        raw = json.loads(p.read_text(encoding="utf-8"))
        if "config" in raw:
            self.config = CalibrationConfig.from_dict(raw["config"])

        # Restore isotonic model
        if "isotonic_state" in raw and raw["isotonic_state"]:
            try:
                import numpy as np
                from sklearn.isotonic import IsotonicRegression  # lazy

                ir = IsotonicRegression(out_of_bounds="clip")
                state = raw["isotonic_state"]
                for attr in ("X_thresholds_", "y_thresholds_"):
                    if attr in state:
                        setattr(ir, attr, np.array(state[attr],
                                                   dtype=np.float64))
                for attr in ("X_min_", "X_max_"):
                    if attr in state:
                        setattr(ir, attr, float(state[attr]))
                # Mark as fitted
                ir.increasing_ = True
                self._isotonic_model = ir
                logger.info("Isotonic model restored from %s", path)
            except ImportError:
                logger.warning(
                    "sklearn/numpy not available; isotonic state skipped."
                )
            except Exception as exc:
                logger.warning("Failed to restore isotonic state: %s", exc)

        logger.info("Calibration parameters loaded from %s", path)


# ---------------------------------------------------------------------------
# Entity-level helper
# ---------------------------------------------------------------------------


def calibrate_entity_confidence(
    entities: List[Dict[str, Any]],
    calibrator: ConfidenceCalibrator,
) -> List[Dict[str, Any]]:
    """Apply calibration to a list of extracted entity dicts.

    Each entity dict is expected to have a ``"confidence"`` key.  The
    function returns new dicts (copies) with ``"confidence"`` replaced
    by the calibrated value and the original stored under
    ``"raw_confidence"``.

    When the calibration method is ``NONE``, entities are returned as
    shallow copies with no transformation.

    Args:
        entities:   List of entity dicts from the extraction pipeline.
        calibrator: A configured :class:`ConfidenceCalibrator`.

    Returns:
        List of entity dicts with calibrated confidence values.
    """
    if not entities:
        return []

    result: List[Dict[str, Any]] = []
    for entity in entities:
        out = dict(entity)  # shallow copy
        raw_conf = float(out.get("confidence", 0.0))

        if calibrator.config.method == CalibrationMethod.NONE:
            result.append(out)
            continue

        # Calibrate as a single-element vector and take the first value
        calibrated = calibrator.calibrate([raw_conf])
        out["raw_confidence"] = raw_conf
        out["confidence"] = calibrated[0] if calibrated else raw_conf
        result.append(out)

    return result


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def compute_ece(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the weighted average gap between predicted confidence
    and actual accuracy across *n_bins* equal-width bins in [0, 1].

    A perfectly calibrated model yields ECE = 0.0.

    Args:
        predictions: Predicted confidence scores (0.0–1.0).
        labels:      Ground-truth binary labels (0 or 1).
        n_bins:      Number of equal-width calibration bins.

    Returns:
        ECE as a float in [0.0, 1.0].
    """
    if not predictions or not labels:
        return 0.0

    n = min(len(predictions), len(labels))
    predictions = predictions[:n]
    labels = labels[:n]

    # Build bins
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for idx in range(n):
        p = max(0.0, min(1.0, predictions[idx]))
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append(idx)

    ece = 0.0
    for bin_indices in bins:
        if not bin_indices:
            continue
        bin_confs = [predictions[i] for i in bin_indices]
        bin_labels = [labels[i] for i in bin_indices]
        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_labels) / len(bin_labels)
        ece += (len(bin_indices) / n) * abs(avg_acc - avg_conf)

    return ece


def compute_reliability_diagram(
    predictions: List[float],
    labels: List[int],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute bin-level data for a reliability diagram.

    Returns a dict with:
    - ``bins``: list of dicts, each with ``bin_start``, ``bin_end``,
      ``avg_confidence``, ``avg_accuracy``, ``count``.
    - ``ece``: the overall ECE value.
    - ``n_bins``: number of bins used.
    - ``n_samples``: total number of samples.

    Args:
        predictions: Predicted confidence scores (0.0–1.0).
        labels:      Ground-truth binary labels (0 or 1).
        n_bins:      Number of equal-width calibration bins.

    Returns:
        Reliability diagram data dict.
    """
    if not predictions or not labels:
        return {
            "bins": [],
            "ece": 0.0,
            "n_bins": n_bins,
            "n_samples": 0,
        }

    n = min(len(predictions), len(labels))
    predictions = predictions[:n]
    labels = labels[:n]

    # Build bins
    bin_data: List[Dict[str, Any]] = []
    bin_width = 1.0 / n_bins

    all_bins: List[List[int]] = [[] for _ in range(n_bins)]
    for idx in range(n):
        p = max(0.0, min(1.0, predictions[idx]))
        bin_idx = min(int(p * n_bins), n_bins - 1)
        all_bins[bin_idx].append(idx)

    for b in range(n_bins):
        bin_start = b * bin_width
        bin_end = (b + 1) * bin_width
        indices = all_bins[b]
        if indices:
            avg_conf = sum(predictions[i] for i in indices) / len(indices)
            avg_acc = sum(labels[i] for i in indices) / len(indices)
        else:
            avg_conf = 0.0
            avg_acc = 0.0
        bin_data.append({
            "bin_start": round(bin_start, 4),
            "bin_end": round(bin_end, 4),
            "avg_confidence": round(avg_conf, 6),
            "avg_accuracy": round(avg_acc, 6),
            "count": len(indices),
        })

    ece = compute_ece(predictions, labels, n_bins)

    return {
        "bins": bin_data,
        "ece": round(ece, 6),
        "n_bins": n_bins,
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Module-level convenience: default calibrator from env vars
# ---------------------------------------------------------------------------


def get_default_calibrator() -> ConfidenceCalibrator:
    """Build a :class:`ConfidenceCalibrator` from environment variables.

    Uses ``LAYOUTLM_CALIBRATION_METHOD`` and ``LAYOUTLM_CALIBRATION_PATH``.
    """
    method = _resolve_method(LAYOUTLM_CALIBRATION_METHOD)
    config = CalibrationConfig(method=method)
    calibrator = ConfidenceCalibrator(config)

    if LAYOUTLM_CALIBRATION_PATH:
        try:
            calibrator.load(LAYOUTLM_CALIBRATION_PATH)
        except Exception as exc:
            logger.warning(
                "Failed to load calibration from %s: %s",
                LAYOUTLM_CALIBRATION_PATH,
                exc,
            )

    return calibrator
