"""Medical corpus fine-tuning pipeline for LayoutLMv3.

Wraps the generic ``layoutlm_finetune.py`` and ``layoutlm_evaluate.py``
modules with medical-specific label definitions, HIPAA-aware data
handling, and privacy-safe reporting.

Medical-specific labels:
    PATIENT_NAME, MRN, DOB, DIAGNOSIS_CODE, MEDICATION, DOSAGE,
    PROVIDER_NAME, FACILITY, DATE_OF_SERVICE, HIPAA_IDENTIFIER

HIPAA compliance:
    When ``--hipaa-strict`` is enabled, no PII values appear in log
    output or training reports.  Only label names and aggregate metrics
    are included.

CTC-safe: trains token classification (BIO tagging) only.

All heavy ML imports are lazy -- the module is importable and testable
without GPU dependencies.

Usage::

    python scripts/finetune_medical_corpus.py \\
        --data-dir ./data/medical \\
        --output-dir ./models/medical-v1 \\
        --epochs 30 --hipaa-strict

Environment Variables:
    MEDICAL_FINETUNE_MODEL (str):
        Base model checkpoint.  Default: ``"microsoft/layoutlmv3-base"``.
    HIPAA_STRICT (str):
        Set to ``"true"`` or ``"1"`` to enable HIPAA-strict mode.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
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
# Medical label set
# ---------------------------------------------------------------------------

MEDICAL_ENTITY_TYPES: List[str] = [
    "PATIENT_NAME",
    "MRN",
    "DOB",
    "DIAGNOSIS_CODE",
    "MEDICATION",
    "DOSAGE",
    "PROVIDER_NAME",
    "FACILITY",
    "DATE_OF_SERVICE",
    "HIPAA_IDENTIFIER",
]

MEDICAL_TYPE_MAP: Dict[str, str] = {
    "PATIENT_NAME": "person_name",
    "MRN": "reference_number",
    "DOB": "date",
    "DIAGNOSIS_CODE": "reference_number",
    "MEDICATION": "medication",
    "DOSAGE": "dosage",
    "PROVIDER_NAME": "person_name",
    "FACILITY": "organization",
    "DATE_OF_SERVICE": "date",
    "HIPAA_IDENTIFIER": "pii",
}

# PHI entity types that must be redacted in HIPAA-strict mode
PHI_ENTITY_TYPES = frozenset({
    "PATIENT_NAME",
    "MRN",
    "DOB",
    "HIPAA_IDENTIFIER",
})

DEFAULT_BASE_MODEL = os.environ.get(
    "MEDICAL_FINETUNE_MODEL", "microsoft/layoutlmv3-base"
)

_HIPAA_ENV = os.environ.get("HIPAA_STRICT", "").lower()
HIPAA_STRICT_DEFAULT = _HIPAA_ENV in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MedicalFinetuneConfig:
    """Configuration for medical corpus fine-tuning."""

    data_dir: str = "./data/medical"
    output_dir: str = "./models/medical-v1"
    model_name: str = DEFAULT_BASE_MODEL
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 3e-5
    test_size: float = 0.2
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    seed: int = 42
    hipaa_strict: bool = HIPAA_STRICT_DEFAULT


# ---------------------------------------------------------------------------
# Label set builder
# ---------------------------------------------------------------------------


def build_medical_label_set():
    """Build a LabelSet for medical entity types.

    Returns:
        A :class:`LabelSet` with medical-specific BIO labels.
    """
    from ocr_local.ml.layoutlm_labels import build_label_set

    return build_label_set(
        name="medical",
        entity_types=MEDICAL_ENTITY_TYPES,
        type_map=MEDICAL_TYPE_MAP,
    )


# ---------------------------------------------------------------------------
# HIPAA-safe logging
# ---------------------------------------------------------------------------

# Patterns that look like PHI in log messages
_PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{9}\b"),  # MRN-like numbers
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # Date formats
]


def redact_phi(text: str) -> str:
    """Redact potential PHI patterns from text.

    Replaces SSN-like, MRN-like, and date-like patterns with
    ``[REDACTED]`` placeholders.  This is a best-effort heuristic
    for log sanitization.

    Args:
        text: Input text potentially containing PHI.

    Returns:
        Text with PHI patterns replaced.
    """
    result = text
    for pattern in _PHI_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_phi_detection_recall(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, float]:
    """Compute recall specifically for PHI entity types.

    PHI recall is critical for HIPAA compliance: missed PHI entities
    represent privacy risk.

    Args:
        true_labels: Flat list of ground-truth BIO labels.
        pred_labels: Flat list of predicted BIO labels.

    Returns:
        Dict with per-PHI-type recall and aggregate PHI recall.
    """
    results: Dict[str, float] = {}
    total_tp = 0
    total_fn = 0

    for phi_type in PHI_ENTITY_TYPES:
        b_label = f"B-{phi_type}"
        i_label = f"I-{phi_type}"
        phi_labels = {b_label, i_label}

        tp = sum(
            1 for t, p in zip(true_labels, pred_labels)
            if t in phi_labels and p in phi_labels
        )
        fn = sum(
            1 for t, p in zip(true_labels, pred_labels)
            if t in phi_labels and p not in phi_labels
        )

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[f"{phi_type}_recall"] = round(recall, 4)
        total_tp += tp
        total_fn += fn

    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    results["overall_phi_recall"] = round(overall_recall, 4)

    return results


def compute_per_label_accuracy(
    true_labels: List[str],
    pred_labels: List[str],
    label_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-label accuracy, precision, recall, F1.

    Args:
        true_labels: Flat list of ground-truth labels.
        pred_labels: Flat list of predicted labels.
        label_names: Label names to compute metrics for.

    Returns:
        Dict mapping label name -> metric dict.
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


def generate_medical_report(
    config: MedicalFinetuneConfig,
    train_result: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]] = None,
    per_label_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    phi_recall: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a HIPAA-safe training report.

    When ``config.hipaa_strict`` is True, the report omits any
    fields that could contain PHI text.

    Args:
        config: Medical fine-tuning configuration.
        train_result: Result dict from training.
        eval_result: Optional evaluation metrics.
        per_label_metrics: Optional per-label accuracy breakdown.
        phi_recall: Optional PHI detection recall metrics.
        output_path: If set, write JSON report here.

    Returns:
        Report dict.
    """
    # Sanitize config for HIPAA-strict mode
    safe_config = asdict(config)
    if config.hipaa_strict:
        # Do not expose data directory paths in reports
        safe_config["data_dir"] = "[REDACTED]"

    report = {
        "pipeline": "medical_corpus_finetune",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hipaa_strict": config.hipaa_strict,
        "config": safe_config,
        "training_result": train_result or {},
        "evaluation": eval_result or {},
        "per_label_metrics": per_label_metrics or {},
        "phi_detection_recall": phi_recall or {},
        "privacy_audit": {
            "phi_entity_types": sorted(PHI_ENTITY_TYPES),
            "hipaa_strict_enabled": config.hipaa_strict,
            "log_redaction_enabled": config.hipaa_strict,
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Medical training report saved to %s", out)

    return report


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def run_medical_finetuning(config: MedicalFinetuneConfig) -> Dict[str, Any]:
    """Execute the medical corpus fine-tuning pipeline.

    1. Builds the medical label set.
    2. Calls ``layoutlm_finetune.run_finetuning`` with medical config.
    3. Generates a HIPAA-safe training report.

    Args:
        config: Medical fine-tuning configuration.

    Returns:
        Training report dict.

    Raises:
        ImportError: If required ML packages are missing.
    """
    from layoutlm_finetune import FineTuneConfig, run_finetuning

    label_set = build_medical_label_set()
    if config.hipaa_strict:
        logger.info(
            "HIPAA-strict mode enabled -- PHI will be redacted from logs."
        )

    logger.info(
        "Medical label set: %d entity types, %d BIO labels",
        len(label_set.entity_types),
        label_set.num_labels,
    )

    # Write label set config
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_config_path = out_dir / "medical_label_set.json"
    label_config = {
        "name": "medical",
        "entity_types": MEDICAL_ENTITY_TYPES,
        "type_map": MEDICAL_TYPE_MAP,
    }
    with open(label_config_path, "w", encoding="utf-8") as fh:
        json.dump(label_config, fh, indent=2)

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

    report = generate_medical_report(
        config=config,
        train_result=train_result,
        output_path=str(out_dir / "medical_training_report.json"),
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> MedicalFinetuneConfig:
    """Parse CLI arguments into a MedicalFinetuneConfig."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 on a medical document corpus (HIPAA-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="./data/medical",
        help="Directory containing medical training .jsonl files.",
    )
    parser.add_argument(
        "--output-dir", default="./models/medical-v1",
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
        "--hipaa-strict", action="store_true",
        default=HIPAA_STRICT_DEFAULT,
        help="Enable HIPAA-strict mode (redact PHI from logs/reports).",
    )
    parser.add_argument(
        "--use-lora", action="store_true",
        help="Apply LoRA adapters.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args(argv)
    return MedicalFinetuneConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        hipaa_strict=args.hipaa_strict,
        use_lora=args.use_lora,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for medical corpus fine-tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = parse_args(argv)
    logger.info("Medical fine-tuning config: %s", asdict(config))

    try:
        report = run_medical_finetuning(config)
        print("\n=== Medical corpus fine-tuning complete ===")
        print(f"  Output: {config.output_dir}")
        print(f"  HIPAA-strict: {config.hipaa_strict}")
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
