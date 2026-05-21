"""LayoutLMv3 fine-tuning script for domain-specific token classification.

CLI entry-point for training a LayoutLMv3 model on custom annotated data
using HuggingFace Trainer.  Supports optional LoRA adapters via the
``peft`` library for parameter-efficient fine-tuning.

CTC-safe: this script trains **token classification** (BIO tagging)
models only — no text generation, no causal LM heads.

All heavy ML imports (``torch``, ``transformers``, ``peft``, ``datasets``,
``seqeval``) are **lazy** — imported inside functions, not at module level
— so the module can be imported and tested without GPU dependencies.

Usage::

    python layoutlm_finetune.py \\
        --dataset custom \\
        --data-dir ./data \\
        --output-dir ./models/out \\
        --label-set forensic \\
        --epochs 50 \\
        --batch-size 2 \\
        --learning-rate 5e-5 \\
        --use-lora --lora-rank 16 --lora-alpha 32

Environment Variables:
    LAYOUTLM_FINETUNE_MODEL (str):
        Base model checkpoint.  Default: ``"microsoft/layoutlmv3-base"``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ocr_local.ml.layoutlm_labels import LabelSet, load_label_set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = os.environ.get(
    "LAYOUTLM_FINETUNE_MODEL", "microsoft/layoutlmv3-base"
)


@dataclass
class FineTuneConfig:
    """Configuration for a LayoutLMv3 fine-tuning run.

    Attributes:
        dataset:        Dataset format (currently ``"custom"`` JSONL).
        data_dir:       Directory containing training JSONL files.
        output_dir:     Directory to save the trained model.
        label_set:      Name of the label set (built-in or path to JSON).
        base_model:     HuggingFace model checkpoint to fine-tune.
        use_lora:       Whether to apply LoRA adapters.
        lora_rank:      LoRA rank (``r`` parameter).
        lora_alpha:     LoRA alpha scaling factor.
        epochs:         Number of training epochs.
        batch_size:     Per-device training batch size.
        learning_rate:  Peak learning rate.
        test_size:      Fraction of data for evaluation split.
        seed:           Random seed for reproducibility.
    """

    dataset: str = "custom"
    data_dir: str = "./data"
    output_dir: str = "./models/out"
    label_set: str = "default"
    base_model: str = DEFAULT_BASE_MODEL
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 5e-5
    test_size: float = 0.2
    seed: int = 42


# ---------------------------------------------------------------------------
# Seqeval metric computation
# ---------------------------------------------------------------------------


def _build_compute_metrics(label_set: LabelSet):
    """Return a ``compute_metrics`` callable for HuggingFace Trainer.

    Uses ``seqeval`` for entity-level (not token-level) evaluation.
    The returned function accepts a Trainer ``EvalPrediction`` namedtuple
    and returns a dict of metric values.

    Args:
        label_set: The :class:`LabelSet` used for id-to-label mapping.

    Returns:
        A callable ``compute_metrics(eval_pred) -> dict``.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for metric computation. "
            "Install with: pip install numpy"
        ) from exc

    try:
        from seqeval.metrics import (
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError(
            "seqeval is required for evaluation metrics. "
            "Install with: pip install seqeval"
        ) from exc

    id2label = label_set.id2label

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels: List[List[str]] = []
        pred_labels: List[List[str]] = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq: List[str] = []
            pred_seq_str: List[str] = []
            for p, lbl in zip(pred_seq, label_seq):
                if lbl == -100:
                    continue
                true_seq.append(id2label.get(int(lbl), "O"))
                pred_seq_str.append(id2label.get(int(p), "O"))
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_str)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Fine-tuning entry point
# ---------------------------------------------------------------------------


def run_finetuning(config: FineTuneConfig) -> Dict[str, Any]:
    """Execute a LayoutLMv3 fine-tuning run.

    Loads data from *config.data_dir*, trains a
    :class:`LayoutLMv3ForTokenClassification` model (optionally with LoRA),
    evaluates at each epoch using seqeval, and saves the final model plus
    metadata to *config.output_dir*.

    Args:
        config: A :class:`FineTuneConfig` with all training parameters.

    Returns:
        Dict with training metrics (``"train_loss"``, ``"eval_f1"``, etc.)
        and metadata (``"model_path"``, ``"label_set"``, ``"duration_s"``).

    Raises:
        ImportError: If required ML packages are missing.
        FileNotFoundError: If data directory or JSONL files are missing.
    """
    # -- Lazy imports ---------------------------------------------------------
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch is required for fine-tuning. "
            "Install with: pip install torch"
        ) from exc

    try:
        from transformers import (
            LayoutLMv3ForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers is required for fine-tuning. "
            "Install with: pip install transformers"
        ) from exc

    from layoutlm_data import create_hf_dataset, load_custom_jsonl

    # -- Load label set -------------------------------------------------------
    ls = load_label_set(config.label_set)
    logger.info(
        "Label set %r: %d labels, %d entity types",
        ls.name, ls.num_labels, len(ls.entity_types),
    )

    # -- Load data ------------------------------------------------------------
    data_dir = Path(config.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No .jsonl files found in {data_dir}"
        )

    all_pages = []
    for jf in jsonl_files:
        pages = load_custom_jsonl(str(jf), ls)
        all_pages.extend(pages)
        logger.info("Loaded %d pages from %s", len(pages), jf.name)

    logger.info("Total annotated pages: %d", len(all_pages))

    # -- Create HF dataset ----------------------------------------------------
    ds = create_hf_dataset(
        all_pages, ls, config.base_model,
        test_size=config.test_size, seed=config.seed,
    )
    logger.info(
        "Dataset: train=%d, test=%d",
        len(ds["train"]), len(ds["test"]),
    )

    # -- Model ----------------------------------------------------------------
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=ls.num_labels,
        id2label=ls.id2label,
        label2id=ls.label2id,
    )

    # -- Optional LoRA --------------------------------------------------------
    if config.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA fine-tuning. "
                "Install with: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            "LoRA applied: rank=%d, alpha=%d, targets=[query, value]",
            config.lora_rank, config.lora_alpha,
        )

    # -- Training arguments ---------------------------------------------------
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        seed=config.seed,
        remove_unused_columns=False,
    )

    # -- Trainer --------------------------------------------------------------
    compute_metrics = _build_compute_metrics(ls)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    start_time = datetime.datetime.now(datetime.timezone.utc)
    train_result = trainer.train()
    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration_s = (end_time - start_time).total_seconds()

    # -- Evaluate -------------------------------------------------------------
    eval_result = trainer.evaluate()

    # -- Save model -----------------------------------------------------------
    trainer.save_model(str(output_dir))

    # Save label set
    label_set_path = output_dir / "label_set.json"
    label_set_data = {
        "name": ls.name,
        "entity_types": list(ls.entity_types),
        "bio_labels": list(ls.bio_labels),
        "label2id": ls.label2id,
        "id2label": {str(k): v for k, v in ls.id2label.items()},
        "type_map": ls.type_map,
    }
    with open(label_set_path, "w", encoding="utf-8") as fh:
        json.dump(label_set_data, fh, indent=2)

    # Save training metadata
    metadata = {
        "base_model": config.base_model,
        "label_set": ls.name,
        "num_labels": ls.num_labels,
        "use_lora": config.use_lora,
        "lora_rank": config.lora_rank if config.use_lora else None,
        "lora_alpha": config.lora_alpha if config.use_lora else None,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "train_samples": len(ds["train"]),
        "test_samples": len(ds["test"]),
        "duration_seconds": duration_s,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_result,
    }
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Model saved to %s", output_dir)
    logger.info(
        "Training complete — loss=%.4f, eval_f1=%.4f, duration=%.1fs",
        train_result.training_loss,
        eval_result.get("eval_f1", 0.0),
        duration_s,
    )

    return {
        "model_path": str(output_dir),
        "label_set": ls.name,
        "train_loss": train_result.training_loss,
        "eval_f1": eval_result.get("eval_f1", 0.0),
        "eval_precision": eval_result.get("eval_precision", 0.0),
        "eval_recall": eval_result.get("eval_recall", 0.0),
        "duration_s": duration_s,
        "epochs": config.epochs,
        "use_lora": config.use_lora,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> FineTuneConfig:
    """Parse command-line arguments into a :class:`FineTuneConfig`.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Populated :class:`FineTuneConfig`.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 for token classification (BIO tagging).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="custom",
        help="Dataset format (currently 'custom' JSONL).",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Directory containing training .jsonl files.",
    )
    parser.add_argument(
        "--output-dir", default="./models/out",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--label-set", default="default",
        help="Label set name (built-in) or path to .json config.",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help="HuggingFace model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--use-lora", action="store_true",
        help="Apply LoRA adapters for parameter-efficient training.",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (r parameter).",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to use for evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args(argv)
    return FineTuneConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        label_set=args.label_set,
        base_model=args.base_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point for LayoutLMv3 fine-tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = _parse_args(argv)
    logger.info("Fine-tuning config: %s", asdict(config))

    result = run_finetuning(config)

    print("\n=== Fine-tuning complete ===")
    for key, val in result.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
