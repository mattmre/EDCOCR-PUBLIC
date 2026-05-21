"""Multi-label document classification with customer-specific taxonomies.

Extends the existing classification system (classification.py) to support
custom taxonomies loaded from JSON config files and multi-label assignment
where a document can match multiple categories simultaneously.

This module is opt-in via the ENABLE_MULTI_LABEL_CLASSIFICATION env var.
Custom taxonomies are loaded from the path specified by
CLASSIFICATION_TAXONOMY_PATH.

Output is merged into the existing classification JSON under a
"multi_label_results" key, preserving full backward compatibility.

Usage:
    from multi_label_classification import MultiLabelClassifier

    classifier = MultiLabelClassifier(config_path="/app/taxonomies.json")
    results = classifier.classify("Invoice for HR department onboarding")
    # results == {"taxonomies": {"department": [{"label": "HR", ...}, ...]}}
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-driven, opt-in)
# ---------------------------------------------------------------------------

ENABLE_MULTI_LABEL_CLASSIFICATION = os.environ.get(
    "ENABLE_MULTI_LABEL_CLASSIFICATION", ""
).lower() in ("1", "true", "yes")

CLASSIFICATION_TAXONOMY_PATH = os.environ.get(
    "CLASSIFICATION_TAXONOMY_PATH", ""
).strip()

# Safety limits
_MAX_TAXONOMIES = 50
_MAX_LABELS_PER_TAXONOMY = 200
_MAX_KEYWORDS_PER_LABEL = 500
_MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CustomTaxonomy:
    """User-defined classification taxonomy.

    Attributes:
        name: Unique taxonomy identifier (e.g. "department", "priority").
        labels: List of valid label names within this taxonomy.
        rules: Mapping of label -> list of keyword/regex patterns.
        exclusive: If True, only the top-scoring label is returned.
                   If False (default), all labels above threshold are returned.
        confidence_threshold: Minimum score for a label to be included.
    """

    name: str
    labels: list = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    exclusive: bool = False
    confidence_threshold: float = 0.5


@dataclass
class TaxonomyMatch:
    """A single label match within a taxonomy.

    Attributes:
        label: The matched label name.
        confidence: Score between 0.0 and 1.0.
        matched_keywords: Number of keyword patterns that matched.
        total_keywords: Total keyword patterns checked for this label.
    """

    label: str
    confidence: float = 0.0
    matched_keywords: int = 0
    total_keywords: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "matched_keywords": self.matched_keywords,
            "total_keywords": self.total_keywords,
        }


# ---------------------------------------------------------------------------
# Taxonomy loading and validation
# ---------------------------------------------------------------------------


def _validate_taxonomy_name(name: str, index: int) -> Optional[str]:
    """Validate and normalize a taxonomy name."""
    if not isinstance(name, str):
        logger.warning(
            "Skipping taxonomy at index %d: name must be a string", index
        )
        return None
    cleaned = name.strip()
    if not cleaned:
        logger.warning("Skipping taxonomy at index %d: empty name", index)
        return None
    if len(cleaned) > 256:
        logger.warning(
            "Skipping taxonomy at index %d: name exceeds 256 chars", index
        )
        return None
    return cleaned


def _validate_labels(labels, taxonomy_name: str) -> list:
    """Validate and normalize a list of label names."""
    if not isinstance(labels, list):
        logger.warning(
            "Taxonomy %s: labels must be a list, got %s",
            taxonomy_name,
            type(labels).__name__,
        )
        return []
    result = []
    for label in labels:
        if not isinstance(label, str):
            continue
        cleaned = label.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    if len(result) > _MAX_LABELS_PER_TAXONOMY:
        logger.warning(
            "Taxonomy %s: truncating labels from %d to %d",
            taxonomy_name,
            len(result),
            _MAX_LABELS_PER_TAXONOMY,
        )
        result = result[:_MAX_LABELS_PER_TAXONOMY]
    return result


def _compile_keyword_patterns(
    keywords: list, label: str, taxonomy_name: str
) -> list:
    """Compile keyword strings into regex patterns."""
    compiled = []
    for kw in keywords:
        if not isinstance(kw, str) or not kw.strip():
            continue
        try:
            compiled.append(re.compile(re.escape(kw.strip()), re.IGNORECASE))
        except re.error as exc:
            logger.warning(
                "Taxonomy %s, label %s: invalid pattern %r: %s",
                taxonomy_name,
                label,
                kw,
                exc,
            )
    return compiled


def _validate_rules(
    rules, labels: list, taxonomy_name: str
) -> dict:
    """Validate and compile rules mapping labels to keyword lists."""
    if not isinstance(rules, dict):
        logger.warning(
            "Taxonomy %s: rules must be a dict, got %s",
            taxonomy_name,
            type(rules).__name__,
        )
        return {}
    result = {}
    label_set = set(labels) if labels else set()
    for label, keywords in rules.items():
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        if label_set and label not in label_set:
            logger.warning(
                "Taxonomy %s: rule label %r not in labels list, skipping",
                taxonomy_name,
                label,
            )
            continue
        if not isinstance(keywords, list):
            logger.warning(
                "Taxonomy %s: keywords for label %r must be a list",
                taxonomy_name,
                label,
            )
            continue
        if len(keywords) > _MAX_KEYWORDS_PER_LABEL:
            logger.warning(
                "Taxonomy %s, label %s: truncating keywords from %d to %d",
                taxonomy_name,
                label,
                len(keywords),
                _MAX_KEYWORDS_PER_LABEL,
            )
            keywords = keywords[:_MAX_KEYWORDS_PER_LABEL]
        compiled = _compile_keyword_patterns(keywords, label, taxonomy_name)
        if compiled:
            result[label] = compiled
    return result


def _parse_taxonomy(raw: dict, index: int) -> Optional[CustomTaxonomy]:
    """Parse and validate a single taxonomy dict from config."""
    if not isinstance(raw, dict):
        logger.warning(
            "Skipping taxonomy at index %d: expected object, got %s",
            index,
            type(raw).__name__,
        )
        return None

    name = _validate_taxonomy_name(raw.get("name", ""), index)
    if name is None:
        return None

    labels = _validate_labels(raw.get("labels", []), name)
    rules = _validate_rules(raw.get("rules", {}), labels, name)

    if not rules:
        logger.warning(
            "Skipping taxonomy %s: no valid rules after validation", name
        )
        return None

    # For labels that appear in rules but not in the labels list, add them
    for label in rules:
        if label not in labels:
            labels.append(label)

    exclusive = bool(raw.get("exclusive", False))

    raw_threshold = raw.get("confidence_threshold", 0.5)
    try:
        threshold = float(raw_threshold)
    except (ValueError, TypeError):
        logger.warning(
            "Taxonomy %s: invalid confidence_threshold %r, using 0.5",
            name,
            raw_threshold,
        )
        threshold = 0.5
    threshold = max(0.0, min(1.0, threshold))

    return CustomTaxonomy(
        name=name,
        labels=labels,
        rules=rules,
        exclusive=exclusive,
        confidence_threshold=threshold,
    )


def load_taxonomies_from_file(config_path: str) -> list:
    """Load custom taxonomies from a JSON configuration file.

    Expected JSON structure::

        {
            "taxonomies": [
                {
                    "name": "department",
                    "labels": ["HR", "Finance", "Legal"],
                    "rules": {
                        "HR": ["employee", "benefits"],
                        "Finance": ["invoice", "payment"]
                    },
                    "exclusive": false,
                    "confidence_threshold": 0.4
                }
            ]
        }

    Args:
        config_path: Absolute or relative path to the JSON config file.

    Returns:
        List of validated CustomTaxonomy objects.
    """
    if not config_path:
        return []

    try:
        file_size = os.path.getsize(config_path)
        if file_size > _MAX_CONFIG_FILE_SIZE:
            logger.error(
                "Taxonomy config file too large (%d bytes, max %d): %s",
                file_size,
                _MAX_CONFIG_FILE_SIZE,
                config_path,
            )
            return []
    except OSError as exc:
        logger.warning("Cannot stat taxonomy config file %s: %s", config_path, exc)
        return []

    try:
        with open(config_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        logger.warning("Taxonomy config file not found: %s", config_path)
        return []
    except json.JSONDecodeError as exc:
        logger.warning("Invalid taxonomy config JSON %s: %s", config_path, exc)
        return []
    except OSError as exc:
        logger.warning("Unable to read taxonomy config file %s: %s", config_path, exc)
        return []

    if not isinstance(payload, dict):
        logger.warning(
            "Invalid taxonomy config payload in %s: expected object", config_path
        )
        return []

    raw_taxonomies = payload.get("taxonomies", [])
    if not isinstance(raw_taxonomies, list):
        logger.warning(
            "Invalid taxonomy config: 'taxonomies' must be a list in %s",
            config_path,
        )
        return []

    if len(raw_taxonomies) > _MAX_TAXONOMIES:
        logger.warning(
            "Truncating taxonomies from %d to %d in %s",
            len(raw_taxonomies),
            _MAX_TAXONOMIES,
            config_path,
        )
        raw_taxonomies = raw_taxonomies[:_MAX_TAXONOMIES]

    taxonomies = []
    seen_names = set()
    for index, raw in enumerate(raw_taxonomies):
        taxonomy = _parse_taxonomy(raw, index)
        if taxonomy is None:
            continue
        if taxonomy.name in seen_names:
            logger.warning(
                "Skipping duplicate taxonomy name %r at index %d",
                taxonomy.name,
                index,
            )
            continue
        seen_names.add(taxonomy.name)
        taxonomies.append(taxonomy)

    logger.info(
        "Loaded %d custom taxonomies from %s", len(taxonomies), config_path
    )
    return taxonomies


# ---------------------------------------------------------------------------
# Multi-label classifier
# ---------------------------------------------------------------------------


class MultiLabelClassifier:
    """Classifier that applies custom taxonomies for multi-label assignment.

    Each taxonomy defines a set of labels with keyword rules. Text is matched
    against these rules, and labels exceeding the confidence threshold are
    returned. In exclusive mode, only the highest-scoring label is returned.

    Thread-safe: instances hold no mutable shared state after construction.

    Args:
        taxonomies: Pre-built list of CustomTaxonomy objects.
        config_path: Path to a JSON config file to load taxonomies from.
            If both taxonomies and config_path are provided, they are merged.
    """

    def __init__(
        self,
        taxonomies: Optional[list] = None,
        config_path: Optional[str] = None,
    ):
        self._taxonomies = list(taxonomies or [])
        if config_path:
            loaded = load_taxonomies_from_file(config_path)
            existing_names = {t.name for t in self._taxonomies}
            for taxonomy in loaded:
                if taxonomy.name not in existing_names:
                    self._taxonomies.append(taxonomy)
                    existing_names.add(taxonomy.name)

    @property
    def taxonomy_names(self) -> list:
        """Return list of loaded taxonomy names."""
        return [t.name for t in self._taxonomies]

    @property
    def taxonomy_count(self) -> int:
        """Return number of loaded taxonomies."""
        return len(self._taxonomies)

    def classify(
        self, text: str, layout_features: Optional[dict] = None
    ) -> dict:
        """Run all loaded taxonomies against the provided text.

        Args:
            text: Document text content to classify.
            layout_features: Optional layout features dict (reserved for
                future use; currently unused).

        Returns:
            Dict with structure::

                {
                    "taxonomies": {
                        "taxonomy_name": [
                            {"label": "...", "confidence": 0.8, ...},
                            ...
                        ],
                        ...
                    },
                    "taxonomy_count": 2,
                    "total_labels_matched": 5
                }
        """
        result = {
            "taxonomies": {t.name: [] for t in self._taxonomies},
            "taxonomy_count": len(self._taxonomies),
            "total_labels_matched": 0,
        }

        if not text or not text.strip():
            return result

        for taxonomy in self._taxonomies:
            matches = self._apply_taxonomy(text, taxonomy)
            result["taxonomies"][taxonomy.name] = [m.to_dict() for m in matches]
            result["total_labels_matched"] += len(matches)

        return result

    def classify_single_taxonomy(
        self, text: str, taxonomy_name: str
    ) -> list:
        """Run a single named taxonomy against the text.

        Args:
            text: Document text content.
            taxonomy_name: Name of the taxonomy to apply.

        Returns:
            List of TaxonomyMatch dicts, or empty list if taxonomy not found.
        """
        for taxonomy in self._taxonomies:
            if taxonomy.name == taxonomy_name:
                matches = self._apply_taxonomy(text, taxonomy)
                return [m.to_dict() for m in matches]
        logger.warning("Taxonomy %r not found", taxonomy_name)
        return []

    def _apply_taxonomy(
        self, text: str, taxonomy: CustomTaxonomy
    ) -> list:
        """Apply a single taxonomy to text and return matching labels.

        Args:
            text: Document text to match against.
            taxonomy: The taxonomy definition to apply.

        Returns:
            List of TaxonomyMatch objects above the confidence threshold.
        """
        if not text or not text.strip():
            return []

        scored = []
        for label, patterns in taxonomy.rules.items():
            if not patterns:
                continue
            hit_count = sum(1 for p in patterns if p.search(text))
            if hit_count == 0:
                continue
            confidence = hit_count / len(patterns)
            scored.append(
                TaxonomyMatch(
                    label=label,
                    confidence=confidence,
                    matched_keywords=hit_count,
                    total_keywords=len(patterns),
                )
            )

        # Sort by confidence descending, then label name ascending for stability
        scored.sort(key=lambda m: (-m.confidence, m.label))

        # Filter by confidence threshold
        filtered = [
            m for m in scored if m.confidence >= taxonomy.confidence_threshold
        ]

        # In exclusive mode, return only the top match
        if taxonomy.exclusive and filtered:
            return [filtered[0]]

        return filtered


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def merge_multi_label_results(
    classification_report: dict,
    multi_label_results: dict,
) -> dict:
    """Merge multi-label classification results into an existing report.

    Adds a "multi_label_results" key to the classification JSON report
    without modifying existing fields.

    Args:
        classification_report: The existing classification JSON dict
            (as written by write_classification_json).
        multi_label_results: Output from MultiLabelClassifier.classify().

    Returns:
        The classification_report dict with multi_label_results added.
    """
    if not isinstance(classification_report, dict):
        return classification_report
    if not isinstance(multi_label_results, dict):
        return classification_report

    classification_report["multi_label_results"] = multi_label_results
    return classification_report


def create_classifier_from_env() -> Optional[MultiLabelClassifier]:
    """Create a MultiLabelClassifier from environment configuration.

    Returns None if multi-label classification is disabled or no taxonomy
    path is configured.

    Returns:
        MultiLabelClassifier instance or None.
    """
    if not ENABLE_MULTI_LABEL_CLASSIFICATION:
        return None

    if not CLASSIFICATION_TAXONOMY_PATH:
        logger.info(
            "Multi-label classification enabled but no CLASSIFICATION_TAXONOMY_PATH set"
        )
        return None

    classifier = MultiLabelClassifier(config_path=CLASSIFICATION_TAXONOMY_PATH)
    if classifier.taxonomy_count == 0:
        logger.warning(
            "Multi-label classification enabled but no valid taxonomies loaded "
            "from %s",
            CLASSIFICATION_TAXONOMY_PATH,
        )
        return None

    logger.info(
        "Multi-label classifier ready with %d taxonomies: %s",
        classifier.taxonomy_count,
        ", ".join(classifier.taxonomy_names),
    )
    return classifier
