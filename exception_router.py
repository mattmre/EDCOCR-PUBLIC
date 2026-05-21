"""Exception routing rules engine for EDCOCR.

Evaluates processed documents against configurable rules and routes
exceptions to the human-review queue. Rules are based on quality
classification, confidence thresholds, feature detection, and custom patterns.

Opt-in via ENABLE_EXCEPTION_ROUTING=true or --enable-exception-routing CLI flag.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENABLE_EXCEPTION_ROUTING = os.environ.get(
    "ENABLE_EXCEPTION_ROUTING", "false"
).lower in ("1", "true", "yes")

# Configurable thresholds via env vars (wrapped in try/except for robustness)
try:
    REVIEW_CONFIDENCE_THRESHOLD = float(
        os.environ.get("REVIEW_CONFIDENCE_THRESHOLD", "0.5")
    )
except (ValueError, TypeError):
    logger.warning(
        "Invalid REVIEW_CONFIDENCE_THRESHOLD value; using default 0.5"
    )
    REVIEW_CONFIDENCE_THRESHOLD = 0.5

try:
    REVIEW_IMAGE_ONLY_THRESHOLD = int(
        os.environ.get("REVIEW_IMAGE_ONLY_THRESHOLD", "3")
    )
except (ValueError, TypeError):
    logger.warning(
        "Invalid REVIEW_IMAGE_ONLY_THRESHOLD value; using default 3"
    )
    REVIEW_IMAGE_ONLY_THRESHOLD = 3

try:
    CLASSIFICATION_CONFIDENCE_THRESHOLD = float(
        os.environ.get("REVIEW_CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.5")
    )
except (ValueError, TypeError):
    logger.warning(
        "Invalid REVIEW_CLASSIFICATION_CONFIDENCE_THRESHOLD value; using default 0.5"
    )
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# Maximum text length to evaluate against custom regex patterns (ReDoS guard).
_MAX_PATTERN_TEXT_LENGTH = 102400  # 100 KB

# Path for custom routing rules JSON file
CUSTOM_RULES_PATH = os.environ.get("EXCEPTION_ROUTING_RULES_PATH", "")


@dataclass
class RoutingRule:
    """A single routing rule definition."""

    name: str
    reason: str  # Maps to ReviewReason value
    enabled: bool = True
    description: str = ""


@dataclass
class RoutingDecision:
    """Result of routing evaluation."""

    should_route: bool = False
    triggered_rules: list = field(default_factory=list)  # List of rule names
    reasons: list = field(default_factory=list)  # List of ReviewReason values
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


class ExceptionRouter:
    """Evaluates documents against routing rules.

    Rules:
    1. Low confidence: overall_confidence < REVIEW_CONFIDENCE_THRESHOLD
    2. Degraded quality: quality classification is 'degraded' or 'review_required'
    3. Handwriting detected: handwriting analysis found handwritten content
    4. Excessive image-only pages: too many pages without text
    5. Classification uncertain: low classification confidence
    6. Custom patterns: regex matches on extracted text (configurable via JSON file)
    """

    def __init__(self, custom_rules_path: str = ""):
        self.rules = self._build_default_rules
        self.custom_patterns: list[dict] = []
        rules_path = custom_rules_path or CUSTOM_RULES_PATH
        if rules_path and os.path.isfile(rules_path):
            self._load_custom_rules(rules_path)

    def _build_default_rules(self) -> list[RoutingRule]:
        """Build the default set of routing rules."""
        return [
            RoutingRule(
                "low_confidence",
                "low_confidence",
                description=(
                    f"Overall confidence below {REVIEW_CONFIDENCE_THRESHOLD}"
                ),
            ),
            RoutingRule(
                "degraded_quality",
                "degraded_quality",
                description="Quality classified as degraded or review_required",
            ),
            RoutingRule(
                "handwriting_detected",
                "handwriting_detected",
                description="Handwriting content detected in document",
            ),
            RoutingRule(
                "image_only_pages",
                "image_only_pages",
                description=(
                    f"More than {REVIEW_IMAGE_ONLY_THRESHOLD} image-only pages"
                ),
            ),
            RoutingRule(
                "classification_uncertain",
                "classification_uncertain",
                description=(
                    f"Document classification confidence below "
                    f"{CLASSIFICATION_CONFIDENCE_THRESHOLD}"
                ),
            ),
        ]

    @staticmethod
    def _is_safe_regex(pattern: str) -> bool:
        """Reject regex patterns with nested quantifiers that cause catastrophic backtracking.

        Checks for patterns like ``(a+)+``, ``(a*)*``, ``(a+)*``, ``(a*)+``
        which exhibit exponential backtracking on non-matching input.
        """
        nested_quantifier = re.compile(r"\([^)]*[+*][^)]*\)[+*?]")
        return not nested_quantifier.search(pattern)

    def _load_custom_rules(self, path: str) -> None:
        """Load custom pattern rules from JSON file.

        Expected format::

            [
                {
                    "name": "rule_name",
                    "pattern": "regex_pattern",
                    "reason": "manual_flag",
                    "description": "Optional description"
                }
            ]

        Invalid entries are logged and skipped.  Patterns with nested
        quantifiers (potential ReDoS) are rejected.
        """
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, list):
                logger.warning(
                    "Custom rules file %s must contain a JSON array; ignoring",
                    path,
                )
                return

            for entry in raw:
                if not isinstance(entry, dict):
                    logger.warning("Skipping non-dict custom rule entry: %r", entry)
                    continue

                name = entry.get("name", "")
                pattern = entry.get("pattern", "")
                if not name or not pattern:
                    logger.warning(
                        "Skipping custom rule with missing name or pattern: %r",
                        entry,
                    )
                    continue

                # ReDoS guard: reject nested quantifiers
                if not self._is_safe_regex(pattern):
                    logger.warning(
                        "Skipping custom rule %r with unsafe regex pattern "
                        "(nested quantifiers detected): %r",
                        name,
                        pattern,
                    )
                    continue

                # Validate regex compiles
                try:
                    re.compile(pattern)
                except re.error as exc:
                    logger.warning(
                        "Skipping custom rule %r with invalid regex %r: %s",
                        name,
                        pattern,
                        exc,
                    )
                    continue

                self.custom_patterns.append(
                    {
                        "name": str(name),
                        "pattern": str(pattern),
                        "reason": str(entry.get("reason", "manual_flag")),
                        "description": str(entry.get("description", "")),
                    }
                )

                # Also register as a RoutingRule for get_rules listing
                self.rules.append(
                    RoutingRule(
                        name=str(name),
                        reason=str(entry.get("reason", "manual_flag")),
                        description=str(entry.get("description", f"Pattern: {pattern}")),
                    )
                )

            logger.info(
                "Loaded %d custom routing rules from %s",
                len(self.custom_patterns),
                path,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load custom rules from %s: %s", path, exc)

    def evaluate(
        self,
        validation_data: dict | None = None,
        classification_data: dict | None = None,
        handwriting_data: dict | None = None,
        extracted_text: str = "",
    ) -> RoutingDecision:
        """Evaluate a document against all enabled rules.

        Args:
            validation_data: Parsed .validation.json content (the ``quality``
                sub-dict or the full report -- both layouts are supported).
            classification_data: Parsed .classification.json content.
            handwriting_data: Parsed .handwriting.json content (the
                ``document_summary`` sub-dict or the full report).
            extracted_text: Full OCR text for pattern matching.

        Returns:
            RoutingDecision with should_route, triggered_rules, and reasons.
        """
        decision = RoutingDecision

        # Normalize validation_data: support both full report and quality sub-dict
        vd = validation_data or {}
        quality = vd.get("quality", vd)

        # Rule 1: Low confidence
        if self._rule_enabled("low_confidence") and vd:
            confidence = quality.get("overall_confidence", 1.0)
            decision.confidence = confidence
            if confidence < REVIEW_CONFIDENCE_THRESHOLD:
                decision.should_route = True
                decision.triggered_rules.append("low_confidence")
                decision.reasons.append("low_confidence")
                decision.metadata["overall_confidence"] = confidence

        # Rule 2: Quality classification
        if self._rule_enabled("degraded_quality") and vd:
            classification = quality.get("classification", "")
            if classification in ("degraded", "review_required"):
                decision.should_route = True
                decision.triggered_rules.append("degraded_quality")
                decision.reasons.append("degraded_quality")
                decision.metadata["quality_classification"] = classification

        # Rule 3: Handwriting detected
        if self._rule_enabled("handwriting_detected") and handwriting_data:
            hw_summary = handwriting_data.get("document_summary", handwriting_data)
            if hw_summary.get("is_primarily_handwritten", False) or hw_summary.get(
                "handwriting_detected", False
            ):
                decision.should_route = True
                decision.triggered_rules.append("handwriting_detected")
                decision.reasons.append("handwriting_detected")

        # Rule 4: Image-only pages
        if self._rule_enabled("image_only_pages") and vd:
            image_only = quality.get("pages_image_only", 0)
            if image_only > REVIEW_IMAGE_ONLY_THRESHOLD:
                decision.should_route = True
                decision.triggered_rules.append("image_only_pages")
                decision.reasons.append("image_only_pages")
                decision.metadata["image_only_pages"] = image_only

        # Rule 5: Classification uncertain
        if self._rule_enabled("classification_uncertain") and classification_data:
            cls_conf = classification_data.get("confidence", 1.0)
            if cls_conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                decision.should_route = True
                decision.triggered_rules.append("classification_uncertain")
                decision.reasons.append("classification_uncertain")
                decision.metadata["classification_confidence"] = cls_conf

        # Rule 6: Custom patterns
        # Truncate text to guard against ReDoS on very large inputs.
        pattern_text = extracted_text[:_MAX_PATTERN_TEXT_LENGTH] if extracted_text else ""
        if pattern_text:
            for pattern_rule in self.custom_patterns:
                try:
                    if re.search(
                        pattern_rule["pattern"], pattern_text, re.IGNORECASE
                    ):
                        decision.should_route = True
                        decision.triggered_rules.append(pattern_rule["name"])
                        decision.reasons.append(pattern_rule["reason"])
                        decision.metadata[
                            f"pattern_{pattern_rule['name']}"
                        ] = True
                except re.error:
                    logger.warning(
                        "Regex error evaluating custom rule %r; skipping",
                        pattern_rule["name"],
                    )

        return decision

    def _rule_enabled(self, name: str) -> bool:
        """Check whether a named rule is enabled."""
        for rule in self.rules:
            if rule.name == name:
                return rule.enabled
        return False

    def get_rules(self) -> list[dict]:
        """Return all rules and their status."""
        return [
            {
                "name": r.name,
                "reason": r.reason,
                "enabled": r.enabled,
                "description": r.description,
            }
            for r in self.rules
        ]
