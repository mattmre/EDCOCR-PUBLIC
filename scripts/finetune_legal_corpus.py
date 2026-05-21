"""Legal corpus fine-tuning pipeline for LayoutLMv3.

Wraps the generic ``layoutlm_finetune.py`` and ``layoutlm_evaluate.py``
modules with legal-specific label definitions, hyperparameter defaults,
and reporting.  Produces a training report with per-label F1 scores,
confusion matrix summary, and loss curve data.

Legal-specific labels:
    CLAUSE_NUMBER, PARTY_NAME, EFFECTIVE_DATE, JURISDICTION,
    SIGNATURE_BLOCK, EXHIBIT_REF, BATES_NUMBER, PRIVILEGE_MARKER

CTC-safe: trains token classification (BIO tagging) only.

All heavy ML imports (``torch``, ``transformers``) are lazy -- the module
is importable and testable without GPU dependencies.

Usage::

    python scripts/finetune_legal_corpus.py \\
        --data-dir ./data/legal \\
        --output-dir ./models/legal-v1 \\
        --epochs 30 \\
        --learning-rate 3e-5

Environment Variables:
    LEGAL_FINETUNE_MODEL (str):
        Base model checkpoint.  Default: ``"microsoft/layoutlmv3-base"``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legal label set
# ---------------------------------------------------------------------------

LEGAL_ENTITY_TYPES: List[str] = [
    "CLAUSE_NUMBER",
    "PARTY_NAME",
    "EFFECTIVE_DATE",
    "JURISDICTION",
    "SIGNATURE_BLOCK",
    "EXHIBIT_REF",
    "BATES_NUMBER",
    "PRIVILEGE_MARKER",
]

LEGAL_TYPE_MAP: Dict[str, str] = {
    "CLAUSE_NUMBER": "reference_number",
    "PARTY_NAME": "person_name",
    "EFFECTIVE_DATE": "date",
    "JURISDICTION": "jurisdiction",
    "SIGNATURE_BLOCK": "signature",
    "EXHIBIT_REF": "reference_number",
    "BATES_NUMBER": "reference_number",
    "PRIVILEGE_MARKER": "privilege_marker",
}

DEFAULT_BASE_MODEL = os.environ.get(
    "LEGAL_FINETUNE_MODEL", "microsoft/layoutlmv3-base"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LegalFinetuneConfig:
    """Configuration for legal corpus fine-tuning."""

    data_dir: str = "./data/legal"
    output_dir: str = "./models/legal-v1"
    model_name: str = DEFAULT_BASE_MODEL
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 3e-5
    test_size: float = 0.2
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42


# ---------------------------------------------------------------------------
# Label set builder
# ---------------------------------------------------------------------------


def build_legal_label_set():
    """Build a LabelSet for legal entity types.

    Returns:
        A :class:`LabelSet` with legal-specific BIO labels.
    """
    from ocr_local.ml.layoutlm_labels import build_label_set

    return build_label_set(
        name="legal",
        entity_types=LEGAL_ENTITY_TYPES,
        type_map=LEGAL_TYPE_MAP,
    )


# ---------------------------------------------------------------------------
# Confusion matrix helpers
# ---------------------------------------------------------------------------


def compute_confusion_summary(
    true_labels: List[str],
    pred_labels: List[str],
    label_names: List[str],
) -> Dict[str, Dict[str, int]]:
    """Compute a confusion matrix summary as a nested dict.

    Args:
        true_labels: Flat list of ground-truth labels.
        pred_labels: Flat list of predicted labels.
        label_names: Unique label names to track.

    Returns:
        Dict mapping true_label -> {pred_label -> count}.
    """
    matrix: Dict[str, Dict[str, int]] = {}
    for name in label_names:
        matrix[name] = {n: 0 for n in label_names}

    for t, p in zip(true_labels, pred_labels):
        if t in matrix and p in matrix.get(t, {}):
            matrix[t][p] += 1

    return matrix


def compute_per_label_f1(
    true_labels: List[str],
    pred_labels: List[str],
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-label precision, recall, and F1.

    Args:
        true_labels: Flat list of ground-truth labels.
        pred_labels: Flat list of predicted labels.
        label_names: Unique label names to compute metrics for.

    Returns:
        Dict mapping label_name -> {precision, recall, f1, support}.
    """
    results: Dict[str, Dict[str, float]] = {}
    for name in label_names:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == name and p == name)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != name and p == name)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == name and p != name)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return results


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------


def generate_training_report(
    config: LegalFinetuneConfig,
    train_result: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]] = None,
    per_label_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    confusion: Optional[Dict[str, Dict[str, int]]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a structured training report.

    Args:
        config: The fine-tuning configuration used.
        train_result: Result dict from ``run_finetuning``.
        eval_result: Optional evaluation metrics dict.
        per_label_metrics: Optional per-label F1 metrics.
        confusion: Optional confusion matrix summary.
        output_path: If provided, write JSON report to this path.

    Returns:
        Report dict with training metadata, metrics, and per-label breakdown.
    """
    report = {
        "pipeline": "legal_corpus_finetune",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": asdict(config),
        "training_result": train_result or {},
        "evaluation": eval_result or {},
        "per_label_metrics": per_label_metrics or {},
        "confusion_matrix_summary": confusion or {},
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Training report saved to %s", out)

    return report


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def run_legal_finetuning(config: LegalFinetuneConfig) -> Dict[str, Any]:
    """Execute the legal corpus fine-tuning pipeline.

    1. Builds the legal label set.
    2. Calls ``layoutlm_finetune.run_finetuning`` with legal config.
    3. Generates a training report with per-label F1 and confusion matrix.

    Args:
        config: Legal fine-tuning configuration.

    Returns:
        Training report dict.

    Raises:
        ImportError: If required ML packages are missing.
    """
    from layoutlm_finetune import FineTuneConfig, run_finetuning

    label_set = build_legal_label_set()
    logger.info(
        "Legal label set: %d entity types, %d BIO labels",
        len(label_set.entity_types),
        label_set.num_labels,
    )

    # Write label set config to output dir for downstream tools
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_config_path = out_dir / "legal_label_set.json"
    label_config = {
        "name": "legal",
        "entity_types": LEGAL_ENTITY_TYPES,
        "type_map": LEGAL_TYPE_MAP,
    }
    with open(label_config_path, "w", encoding="utf-8") as fh:
        json.dump(label_config, fh, indent=2)

    # Build FineTuneConfig
    ft_config = FineTuneConfig(
        dataset="custom",
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        label_set=str(label_config_path),
        base_model=config.model_name,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        test_size=config.test_size,
        seed=config.seed,
    )

    train_result = run_finetuning(ft_config)

    report = generate_training_report(
        config=config,
        train_result=train_result,
        output_path=str(out_dir / "legal_training_report.json"),
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> LegalFinetuneConfig:
    """Parse CLI arguments into a LegalFinetuneConfig."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 on a legal document corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="./data/legal",
        help="Directory containing legal training .jsonl files.",
    )
    parser.add_argument(
        "--output-dir", default="./models/legal-v1",
        help="Directory to save the trained model and reports.",
    )
    parser.add_argument(
        "--model-name", default=DEFAULT_BASE_MODEL,
        help="HuggingFace model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for evaluation split.",
    )
    parser.add_argument(
        "--use-lora", action="store_true",
        help="Apply LoRA adapters for parameter-efficient training.",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank parameter.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args(argv)
    return LegalFinetuneConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for legal corpus fine-tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = parse_args(argv)
    logger.info("Legal fine-tuning config: %s", asdict(config))

    try:
        report = run_legal_finetuning(config)
        print("\n=== Legal corpus fine-tuning complete ===")
        print(f"  Output: {config.output_dir}")
        if "training_result" in report:
            tr = report["training_result"]
            print(f"  Train loss: {tr.get('train_loss', 'N/A')}")
            print(f"  Eval F1:    {tr.get('eval_f1', 'N/A')}")
    except ImportError as exc:
        logger.error("Missing ML dependency: %s", exc)
        print(f"\nError: {exc}")
        print("Install with: pip install torch transformers seqeval datasets")
        sys.exit(1)


if __name__ == "__main__":
    main()
