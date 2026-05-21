"""LayoutLMv3 evaluation and reporting for token classification models.

Provides entity-level (not token-level) evaluation using ``seqeval``,
with per-entity-type breakdown and micro/macro/weighted averages.
Output is a JSON report compatible with the ``benchmark_results/``
directory format.

CTC-safe: evaluates **token classification** (BIO tagging) only — no
text generation models.

All heavy ML imports (``torch``, ``transformers``, ``seqeval``,
``numpy``) are **lazy** — imported inside functions so the module is
importable and testable without GPU dependencies.

Typical usage::

    from layoutlm_evaluate import evaluate_model

    results = evaluate_model(
        model_path="./models/out",
        test_data=[...],       # list of AnnotatedPage
        label_set=label_set,
        output_path="benchmark_results/eval_forensic.json",
    )
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from ocr_local.ml.layoutlm_labels import LabelSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    test_data: List[Any],
    label_set: LabelSet,
    confidence_threshold: float = 0.5,
    output_path: str = "",
) -> Dict[str, Any]:
    """Evaluate a trained LayoutLMv3 token-classification model.

    Runs inference on *test_data* (list of :class:`AnnotatedPage`),
    computes entity-level precision / recall / F1 via ``seqeval``,
    and optionally writes a JSON report.

    Args:
        model_path:            Path to the saved model directory.
        test_data:             List of :class:`AnnotatedPage` instances.
        label_set:             The :class:`LabelSet` used during training.
        confidence_threshold:  Minimum softmax probability to accept a
                               predicted label (below → ``"O"``).
        output_path:           If non-empty, write the JSON report here.

    Returns:
        Dict with keys ``"overall"``, ``"per_entity"``, ``"metadata"``.

    Raises:
        ImportError: If required ML packages are missing.
    """
    # -- Lazy imports ---------------------------------------------------------
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "numpy is required for evaluation. "
            "Install with: pip install numpy"
        ) from exc

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch is required for evaluation. "
            "Install with: pip install torch"
        ) from exc

    try:
        from transformers import (
            AutoTokenizer,
            LayoutLMv3ForTokenClassification,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers is required for evaluation. "
            "Install with: pip install transformers"
        ) from exc

    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError(
            "seqeval is required for evaluation. "
            "Install with: pip install seqeval"
        ) from exc

    # -- Load model + tokenizer -----------------------------------------------
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    model.eval()

    # Try to load tokenizer from model dir; fall back to base model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

    id2label = label_set.id2label

    # -- Run inference --------------------------------------------------------
    all_true_labels: List[List[str]] = []
    all_pred_labels: List[List[str]] = []

    for page in test_data:
        words = [w.text for w in page.words]
        word_labels = [w.label for w in page.words]

        if not words:
            continue

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoding)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        max_probs = probs.max(dim=-1).values.cpu().numpy()

        word_ids = encoding.word_ids()

        true_seq: List[str] = []
        pred_seq: List[str] = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx == previous_word_idx:
                continue
            # First sub-token of each word
            true_label = (
                word_labels[word_idx]
                if word_idx < len(word_labels)
                else "O"
            )
            pred_id = int(pred_ids[idx])
            pred_prob = float(max_probs[idx])

            if pred_prob < confidence_threshold:
                pred_label = "O"
            else:
                pred_label = id2label.get(pred_id, "O")

            true_seq.append(true_label)
            pred_seq.append(pred_label)
            previous_word_idx = word_idx

        all_true_labels.append(true_seq)
        all_pred_labels.append(pred_seq)

    # -- Compute metrics ------------------------------------------------------
    if not all_true_labels:
        report = _empty_report(label_set, model_path)
        if output_path:
            _write_report(report, output_path)
        return report

    overall_f1 = f1_score(all_true_labels, all_pred_labels)
    overall_precision = precision_score(all_true_labels, all_pred_labels)
    overall_recall = recall_score(all_true_labels, all_pred_labels)

    # Per-entity breakdown via classification_report
    report_str = classification_report(
        all_true_labels, all_pred_labels, output_dict=True,
    )

    # Build per-entity results
    per_entity: Dict[str, Dict[str, float]] = {}
    for entity_type in label_set.entity_types:
        if entity_type in report_str:
            entry = report_str[entity_type]
            per_entity[entity_type] = {
                "precision": round(float(entry.get("precision", 0.0)), 4),
                "recall": round(float(entry.get("recall", 0.0)), 4),
                "f1": round(float(entry.get("f1-score", 0.0)), 4),
                "support": int(entry.get("support", 0)),
            }
        else:
            per_entity[entity_type] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }

    # Extract averages from seqeval report
    averages: Dict[str, Dict[str, float]] = {}
    for avg_key in ("micro avg", "macro avg", "weighted avg"):
        if avg_key in report_str:
            entry = report_str[avg_key]
            averages[avg_key.replace(" ", "_")] = {
                "precision": round(float(entry.get("precision", 0.0)), 4),
                "recall": round(float(entry.get("recall", 0.0)), 4),
                "f1": round(float(entry.get("f1-score", 0.0)), 4),
                "support": int(entry.get("support", 0)),
            }

    report = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "label_set": label_set.name,
        "num_labels": label_set.num_labels,
        "confidence_threshold": confidence_threshold,
        "num_pages_evaluated": len(test_data),
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
        },
        "per_entity": per_entity,
        "averages": averages,
        "metadata": {
            "evaluation_type": "token_classification",
            "metric_library": "seqeval",
            "tagging_scheme": "BIO",
        },
    }

    if output_path:
        _write_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Offline evaluation (from pre-computed predictions)
# ---------------------------------------------------------------------------


def evaluate_predictions(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    label_set: LabelSet,
    output_path: str = "",
) -> Dict[str, Any]:
    """Evaluate pre-computed prediction sequences without a model.

    Useful for offline analysis or when predictions have already been
    serialized. Uses ``seqeval`` for entity-level metrics.

    Args:
        true_labels: Ground-truth BIO label sequences (outer: pages).
        pred_labels: Predicted BIO label sequences (same shape).
        label_set:   The :class:`LabelSet` for entity type enumeration.
        output_path: Optional path to write the JSON report.

    Returns:
        Dict with ``"overall"``, ``"per_entity"``, ``"averages"``.
    """
    if not true_labels or not pred_labels:
        report = _empty_report(label_set, "")
        if output_path:
            _write_report(report, output_path)
        return report

    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError(
            "seqeval is required for evaluation. "
            "Install with: pip install seqeval"
        ) from exc

    overall_f1 = f1_score(true_labels, pred_labels)
    overall_precision = precision_score(true_labels, pred_labels)
    overall_recall = recall_score(true_labels, pred_labels)

    report_dict = classification_report(
        true_labels, pred_labels, output_dict=True,
    )

    per_entity: Dict[str, Dict[str, float]] = {}
    for entity_type in label_set.entity_types:
        if entity_type in report_dict:
            entry = report_dict[entity_type]
            per_entity[entity_type] = {
                "precision": round(float(entry.get("precision", 0.0)), 4),
                "recall": round(float(entry.get("recall", 0.0)), 4),
                "f1": round(float(entry.get("f1-score", 0.0)), 4),
                "support": int(entry.get("support", 0)),
            }
        else:
            per_entity[entity_type] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
            }

    averages: Dict[str, Dict[str, float]] = {}
    for avg_key in ("micro avg", "macro avg", "weighted avg"):
        if avg_key in report_dict:
            entry = report_dict[avg_key]
            averages[avg_key.replace(" ", "_")] = {
                "precision": round(float(entry.get("precision", 0.0)), 4),
                "recall": round(float(entry.get("recall", 0.0)), 4),
                "f1": round(float(entry.get("f1-score", 0.0)), 4),
                "support": int(entry.get("support", 0)),
            }

    report = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": "",
        "label_set": label_set.name,
        "num_labels": label_set.num_labels,
        "num_sequences_evaluated": len(true_labels),
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
        },
        "per_entity": per_entity,
        "averages": averages,
        "metadata": {
            "evaluation_type": "token_classification",
            "metric_library": "seqeval",
            "tagging_scheme": "BIO",
        },
    }

    if output_path:
        _write_report(report, output_path)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_report(label_set: LabelSet, model_path: str) -> Dict[str, Any]:
    """Return a zeroed-out report when there is no data to evaluate."""
    per_entity = {
        et: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        for et in label_set.entity_types
    }
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "label_set": label_set.name,
        "num_labels": label_set.num_labels,
        "num_pages_evaluated": 0,
        "overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "per_entity": per_entity,
        "averages": {},
        "metadata": {
            "evaluation_type": "token_classification",
            "metric_library": "seqeval",
            "tagging_scheme": "BIO",
        },
    }


def _write_report(report: Dict[str, Any], output_path: str) -> None:
    """Write a JSON evaluation report to disk.

    Creates parent directories if they do not exist.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Evaluation report saved to %s", path)
