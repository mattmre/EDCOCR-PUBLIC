"""Training data pipeline for LayoutLMv3 domain fine-tuning.

Provides data structures and loaders for annotated document pages in
JSONL format, plus conversion utilities to produce HuggingFace-
compatible datasets for token classification training.

Design principles:
- Pure Python at module level — ``torch``, ``transformers``, and
  ``datasets`` are imported lazily inside functions so the module is
  importable (and testable) without heavy ML dependencies.
- CTC-safe: only token classification (BIO tagging) data is handled.
- Validation: labels are checked against the active :class:`LabelSet`
  at load time, with warnings for unknown labels.

Typical usage::

    from ocr_local.ml.layoutlm_labels import load_label_set
    from layoutlm_data import load_custom_jsonl, create_hf_dataset

    ls = load_label_set("forensic")
    pages = load_custom_jsonl("data/train.jsonl", ls)
    ds = create_hf_dataset(pages, ls, "microsoft/layoutlmv3-base")
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ocr_local.ml.layoutlm_labels import LabelSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedWord:
    """A single word with its bounding box and BIO label.

    Attributes:
        text:  The word text (e.g. ``"Invoice"``).
        bbox:  Bounding box as ``[x0, y0, x1, y1]`` in pixel coords.
        label: BIO tag string, e.g. ``"B-DATE"`` or ``"O"``.
    """

    text: str
    bbox: List[int] = field(default_factory=list)
    label: str = "O"


@dataclass
class AnnotatedPage:
    """A page of annotated words for token-classification training.

    Attributes:
        words:      List of annotated words on the page.
        image_path: Optional path to the page image (for multimodal input).
        doc_id:     Document identifier string.
        page_num:   1-based page number within the document.
    """

    words: List[AnnotatedWord] = field(default_factory=list)
    image_path: Optional[str] = None
    doc_id: str = ""
    page_num: int = 0


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def load_custom_jsonl(path: str, label_set: LabelSet) -> List[AnnotatedPage]:
    """Load annotated pages from a JSONL file.

    Each line in the file is a JSON object representing a single page::

        {
          "doc_id": "doc_001",
          "page_num": 1,
          "words": [
            {"text": "Invoice", "bbox": [10, 20, 100, 40], "label": "O"},
            {"text": "#12345", "bbox": [105, 20, 200, 40], "label": "B-INVOICE_NUMBER"}
          ],
          "image_path": "images/doc_001_p1.png"   // optional
        }

    Labels are validated against *label_set*.  Words with unknown labels
    are assigned ``"O"`` and a warning is logged.

    Args:
        path:      Path to the ``.jsonl`` file.
        label_set: The :class:`LabelSet` to validate labels against.

    Returns:
        List of :class:`AnnotatedPage` instances.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If a line contains malformed JSON or missing keys.
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    bio_labels_set = set(label_set.bio_labels)
    pages: List[AnnotatedPage] = []
    unknown_labels_warned: set = set()

    with open(filepath, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at line {line_num} in {filepath}: {exc}"
                ) from exc

            doc_id = record.get("doc_id", f"unknown_{line_num}")
            page_num = int(record.get("page_num", 0))
            image_path = record.get("image_path")
            raw_words = record.get("words", [])

            words: List[AnnotatedWord] = []
            for w in raw_words:
                text = str(w.get("text", ""))
                bbox = list(w.get("bbox", [0, 0, 0, 0]))
                label = str(w.get("label", "O"))

                # Validate label
                if label not in bio_labels_set:
                    if label not in unknown_labels_warned:
                        warnings.warn(
                            f"Unknown label {label!r} in {doc_id} page "
                            f"{page_num} — defaulting to 'O'. "
                            f"Valid labels: {sorted(bio_labels_set)}",
                            stacklevel=2,
                        )
                        logger.warning(
                            "Unknown label %r in %s page %d, using 'O'",
                            label, doc_id, page_num,
                        )
                        unknown_labels_warned.add(label)
                    label = "O"

                words.append(AnnotatedWord(text=text, bbox=bbox, label=label))

            pages.append(
                AnnotatedPage(
                    words=words,
                    image_path=image_path,
                    doc_id=doc_id,
                    page_num=page_num,
                )
            )

    logger.info(
        "Loaded %d annotated pages from %s", len(pages), filepath
    )
    return pages


# ---------------------------------------------------------------------------
# HuggingFace dataset creation
# ---------------------------------------------------------------------------


def create_hf_dataset(
    pages: List[AnnotatedPage],
    label_set: LabelSet,
    tokenizer_name: str = "microsoft/layoutlmv3-base",
    test_size: float = 0.2,
    seed: int = 42,
) -> Any:
    """Convert annotated pages into a HuggingFace DatasetDict.

    The returned dict has ``"train"`` and ``"test"`` splits, each
    containing columns suitable for :class:`LayoutLMv3ForTokenClassification`:

    - ``input_ids``, ``attention_mask``, ``bbox`` — tokenised inputs.
    - ``labels`` — integer label IDs aligned to sub-word tokens.

    Sub-word alignment follows the standard convention: the first
    sub-token of each word receives the word's label, and continuation
    sub-tokens receive ``-100`` (ignored by CrossEntropyLoss).

    Args:
        pages:          Annotated pages from :func:`load_custom_jsonl`.
        label_set:      The :class:`LabelSet` defining valid labels.
        tokenizer_name: HuggingFace tokenizer identifier.
        test_size:      Fraction of data for the test split (0.0–1.0).
        seed:           Random seed for the train/test split.

    Returns:
        A ``datasets.DatasetDict`` with ``"train"`` and ``"test"`` keys.

    Raises:
        ImportError: If ``transformers`` or ``datasets`` is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for create_hf_dataset. "
            "Install with: pip install transformers"
        ) from exc

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "datasets is required for create_hf_dataset. "
            "Install with: pip install datasets"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    label2id = label_set.label2id

    all_input_ids = []
    all_attention_masks = []
    all_bboxes = []
    all_labels = []

    for page in pages:
        words = [w.text for w in page.words]
        word_bboxes = [w.bbox for w in page.words]
        word_labels = [w.label for w in page.words]

        if not words:
            continue

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None,
        )

        # Align labels to sub-word tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        aligned_bboxes = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
            elif word_idx != previous_word_idx:
                # First sub-token of a word
                lbl = word_labels[word_idx]
                aligned_labels.append(label2id.get(lbl, 0))
                aligned_bboxes.append(
                    word_bboxes[word_idx]
                    if word_idx < len(word_bboxes)
                    else [0, 0, 0, 0]
                )
            else:
                # Continuation sub-token
                aligned_labels.append(-100)
                aligned_bboxes.append(
                    word_bboxes[word_idx]
                    if word_idx < len(word_bboxes)
                    else [0, 0, 0, 0]
                )
            previous_word_idx = word_idx

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_bboxes.append(aligned_bboxes)
        all_labels.append(aligned_labels)

    dataset_dict: Dict[str, Any] = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "bbox": all_bboxes,
        "labels": all_labels,
    }

    ds = Dataset.from_dict(dataset_dict)

    if test_size > 0.0 and len(ds) > 1:
        split = ds.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({"train": split["train"], "test": split["test"]})

    # Not enough data to split — put everything in train
    return DatasetDict({"train": ds, "test": ds})
