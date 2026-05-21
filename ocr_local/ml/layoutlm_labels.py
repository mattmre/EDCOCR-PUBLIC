"""Single source of truth for LayoutLMv3 entity labels (BIO tagging).

This module centralises every label set used by the LayoutLMv3 token-
classification head.  All other modules (``semantic_extraction``,
``layoutlm_structure``, training scripts) should import from here
rather than defining their own label lists.

CTC-safe: this module is used exclusively for **token classification**
(sequence-labelling / NER) tasks, not text generation.  All labels
follow the BIO (Begin-Inside-Outside) tagging convention.

Design goals:
- Pure Python — no ``torch``, ``transformers``, or heavy-ML imports.
- Deterministic ordering — label indices are stable across runs.
- Extensible — custom label sets can be loaded from JSON files.
- Environment-driven — ``LAYOUTLM_LABEL_SET`` and
  ``LAYOUTLM_LABEL_CONFIG`` env vars steer runtime selection.

Environment Variables:
    LAYOUTLM_LABEL_SET (str):
        Name of a built-in label set to activate.  One of ``"default"``,
        ``"forensic"``, ``"receipt"``, ``"form"``.  Default: ``"default"``.
    LAYOUTLM_LABEL_CONFIG (str):
        Path to a JSON file describing a custom label set.  When set,
        this takes precedence over ``LAYOUTLM_LABEL_SET``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default extraction-type map
# ---------------------------------------------------------------------------

DEFAULT_TYPE_MAP: Dict[str, str] = {
    "INVOICE_NUMBER": "reference_number",
    "DATE": "date",
    "AMOUNT": "amount",
    "PERSON_NAME": "person_name",
    "ORGANIZATION": "organization",
    "ADDRESS": "address",
    "REFERENCE_NUMBER": "reference_number",
    "PHONE_NUMBER": "phone_number",
    "EMAIL": "email_address",
    # Forensic
    "CASE_NUMBER": "reference_number",
    "BATES_NUMBER": "reference_number",
    "EXHIBIT_NUMBER": "reference_number",
    "COURT_NAME": "organization",
    "JUDGE_NAME": "person_name",
    "ATTORNEY_NAME": "person_name",
    "FILING_DATE": "date",
    "DOCKET_NUMBER": "reference_number",
    # Receipt
    "STORE_NAME": "organization",
    "STORE_ADDRESS": "address",
    "TIME": "time",
    "ITEM_NAME": "line_item",
    "ITEM_PRICE": "amount",
    "SUBTOTAL": "amount",
    "TAX": "amount",
    "TOTAL": "amount",
    "PAYMENT_METHOD": "payment_method",
    "CARD_NUMBER": "reference_number",
    # Form
    "FIELD_LABEL": "field_label",
    "FIELD_VALUE": "field_value",
    "CHECKBOX": "checkbox",
    "SIGNATURE_FIELD": "signature",
    "DATE_FIELD": "date",
}

# ---------------------------------------------------------------------------
# Built-in label sets (entity-type lists — BIO expansion happens later)
# ---------------------------------------------------------------------------

BUILTIN_LABEL_SETS: Dict[str, List[str]] = {
    "default": [
        "INVOICE_NUMBER",
        "DATE",
        "AMOUNT",
        "PERSON_NAME",
        "ORGANIZATION",
        "ADDRESS",
        "REFERENCE_NUMBER",
        "PHONE_NUMBER",
        "EMAIL",
    ],
    "forensic": [
        "INVOICE_NUMBER",
        "DATE",
        "AMOUNT",
        "PERSON_NAME",
        "ORGANIZATION",
        "ADDRESS",
        "REFERENCE_NUMBER",
        "PHONE_NUMBER",
        "EMAIL",
        "CASE_NUMBER",
        "BATES_NUMBER",
        "EXHIBIT_NUMBER",
        "COURT_NAME",
        "JUDGE_NAME",
        "ATTORNEY_NAME",
        "FILING_DATE",
        "DOCKET_NUMBER",
    ],
    "receipt": [
        "STORE_NAME",
        "STORE_ADDRESS",
        "DATE",
        "TIME",
        "ITEM_NAME",
        "ITEM_PRICE",
        "SUBTOTAL",
        "TAX",
        "TOTAL",
        "PAYMENT_METHOD",
        "CARD_NUMBER",
    ],
    "form": [
        "FIELD_LABEL",
        "FIELD_VALUE",
        "CHECKBOX",
        "SIGNATURE_FIELD",
        "DATE_FIELD",
        "PERSON_NAME",
        "ORGANIZATION",
        "ADDRESS",
    ],
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSet:
    """Immutable descriptor for a LayoutLMv3 BIO label configuration.

    Attributes:
        name:         Human-readable label-set name.
        entity_types: Ordered list of entity type strings (no BIO prefix).
        bio_labels:   Full BIO label list including ``"O"`` sentinel.
        label2id:     Mapping from BIO label string to integer index.
        id2label:     Mapping from integer index to BIO label string.
        type_map:     Mapping from entity type to extraction field type.
        num_labels:   Total number of BIO labels (``len(bio_labels)``).
    """

    name: str
    entity_types: tuple  # tuple for immutability
    bio_labels: tuple  # tuple for immutability
    label2id: Dict[str, int] = field(default_factory=dict)
    id2label: Dict[int, str] = field(default_factory=dict)
    type_map: Dict[str, str] = field(default_factory=dict)
    num_labels: int = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def expand_to_bio(entity_types: List[str]) -> List[str]:
    """Expand a list of entity type names into a full BIO label list.

    The ``"O"`` (Outside) label is always first, followed by ``B-`` / ``I-``
    pairs for each entity type in the order given.

    Args:
        entity_types: Entity type names, e.g. ``["DATE", "AMOUNT"]``.

    Returns:
        A list of BIO label strings, e.g.
        ``["O", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT"]``.

    Examples:
        >>> expand_to_bio(["DATE"])
        ['O', 'B-DATE', 'I-DATE']
        >>> expand_to_bio([])
        ['O']
    """
    labels: List[str] = ["O"]
    for etype in entity_types:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    return labels


def build_label_set(
    name: str,
    entity_types: List[str],
    type_map: Optional[Dict[str, str]] = None,
) -> LabelSet:
    """Build a :class:`LabelSet` from a list of entity type names.

    Args:
        name:         A human-readable identifier for this label set.
        entity_types: Ordered entity type names (without BIO prefix).
        type_map:     Optional mapping from entity type to extraction
                      field type.  Entries from :data:`DEFAULT_TYPE_MAP`
                      are used as fallback for any type not explicitly
                      provided.

    Returns:
        A fully populated :class:`LabelSet` instance.
    """
    bio = expand_to_bio(entity_types)
    l2i = {label: idx for idx, label in enumerate(bio)}
    i2l = {idx: label for idx, label in enumerate(bio)}

    # Merge caller-supplied type_map on top of defaults
    merged_map: Dict[str, str] = {}
    for etype in entity_types:
        if type_map and etype in type_map:
            merged_map[etype] = type_map[etype]
        elif etype in DEFAULT_TYPE_MAP:
            merged_map[etype] = DEFAULT_TYPE_MAP[etype]
        else:
            merged_map[etype] = etype.lower()

    return LabelSet(
        name=name,
        entity_types=tuple(entity_types),
        bio_labels=tuple(bio),
        label2id=l2i,
        id2label=i2l,
        type_map=merged_map,
        num_labels=len(bio),
    )


def load_label_set(name_or_path: str) -> LabelSet:
    """Load a label set by built-in name **or** from a JSON file path.

    If *name_or_path* matches one of the keys in
    :data:`BUILTIN_LABEL_SETS` it is used directly.  Otherwise, if
    the string ends with ``".json"`` it is treated as a file path
    to a custom label-set definition.

    JSON schema::

        {
          "name": "my_custom_set",
          "entity_types": ["FOO", "BAR"],
          "type_map": {"FOO": "foo_field"}   // optional
        }

    Args:
        name_or_path: Built-in set name or path to a ``.json`` file.

    Returns:
        A :class:`LabelSet` for the requested configuration.

    Raises:
        ValueError:  If *name_or_path* is neither a known built-in name
                     nor a path ending in ``".json"``.
        FileNotFoundError: If a JSON path is given but does not exist.
    """
    # Built-in?
    if name_or_path in BUILTIN_LABEL_SETS:
        logger.debug("Loading built-in label set %r", name_or_path)
        return build_label_set(name_or_path, BUILTIN_LABEL_SETS[name_or_path])

    # Custom JSON file?
    if name_or_path.endswith(".json"):
        path = Path(name_or_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Custom label config file not found: {path}"
            )
        logger.info("Loading custom label set from %s", path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        lname = data.get("name", path.stem)
        etypes = data["entity_types"]
        tmap = data.get("type_map")
        return build_label_set(lname, etypes, type_map=tmap)

    raise ValueError(
        f"Unknown label set {name_or_path!r}. Must be one of "
        f"{sorted(BUILTIN_LABEL_SETS)} or a path to a .json file."
    )


def get_active_label_set() -> LabelSet:
    """Return the currently active :class:`LabelSet` based on env vars.

    Resolution order:

    1. If ``LAYOUTLM_LABEL_CONFIG`` is set and points to a ``.json``
       file, that custom config is loaded.
    2. Otherwise ``LAYOUTLM_LABEL_SET`` selects a built-in set
       (default: ``"default"``).

    Returns:
        The active :class:`LabelSet`.
    """
    config_path = os.environ.get("LAYOUTLM_LABEL_CONFIG", "")
    if config_path:
        logger.info(
            "LAYOUTLM_LABEL_CONFIG is set; loading custom labels from %s",
            config_path,
        )
        return load_label_set(config_path)

    set_name = os.environ.get("LAYOUTLM_LABEL_SET", "default")
    logger.debug("Using built-in label set %r", set_name)
    return load_label_set(set_name)
