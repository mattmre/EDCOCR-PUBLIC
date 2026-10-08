"""Exception routing rules engine for OCR_LOCAL.

Evaluates processed documents against configurable rules and routes
exceptions to the human-review queue. Rules are based on quality
classification, confidence thresholds, feature detection, and custom patterns.

Opt-in via ENABLE_EXCEPTION_ROUTING=true or --enable-exception-routing CLI flag.
"""

import json
import logging
import os
import re
import stat
from dataclasses import dataclass, field

from ocr_distributed.text_io import read_text_bounded

logger = logging.getLogger(__name__)

ENABLE_EXCEPTION_ROUTING = os.environ.get(
    "ENABLE_EXCEPTION_ROUTING", "false"
).lower() in ("1", "true", "yes")

# Configurable thresholds via env vars (wrapped in try/except for robustness)
try:
    REVIEW_CONFIDENCE_THRESHOLD = float(
        os.environ.get("REVIEW_CONFIDENCE_THRESHOLD", "0.5")
    )
except (ValueError, TypeError):
    logger.warning("Invalid REVIEW_CONFIDENCE_THRESHOLD value; using default 0.5")
    REVIEW_CONFIDENCE_THRESHOLD = 0.5

try:
    REVIEW_IMAGE_ONLY_THRESHOLD = int(
        os.environ.get("REVIEW_IMAGE_ONLY_THRESHOLD", "3")
    )
except (ValueError, TypeError):
    logger.warning("Invalid REVIEW_IMAGE_ONLY_THRESHOLD value; using default 3")
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
_MAX_CUSTOM_RULES = 100
_MAX_CUSTOM_PATTERN_LENGTH = 2048
_MAX_CUSTOM_NAME_LENGTH = 128
_MAX_CUSTOM_REASON_LENGTH = 128
_MAX_CUSTOM_DESCRIPTION_LENGTH = 1024
_MAX_CUSTOM_RULES_BYTES = 1 * 1024 * 1024

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
        self.rules = self._build_default_rules()
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
                description=(f"Overall confidence below {REVIEW_CONFIDENCE_THRESHOLD}"),
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
        if not isinstance(pattern, str) or len(pattern) > _MAX_CUSTOM_PATTERN_LENGTH:
            return False
        nested_quantifier = re.compile(r"\([^)]*[+*][^)]*\)[+*?]")
        ambiguous_alternation = re.compile(r"\((?:\?:)?[^()]*\|[^()]*\)[+*?]")
        return not (
            nested_quantifier.search(pattern) or ambiguous_alternation.search(pattern)
        )

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
            raw = json.loads(read_text_bounded(path, max_bytes=_MAX_CUSTOM_RULES_BYTES))

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
                if (
                    not isinstance(name, str)
                    or not isinstance(pattern, str)
                    or not name.strip()
                    or not pattern
                ):
                    logger.warning(
                        "Skipping custom rule with invalid name or pattern: %r",
                        entry,
                    )
                    continue

                if len(self.custom_patterns) >= _MAX_CUSTOM_RULES:
                    logger.warning(
                        "Ignoring custom rules after maximum of %d entries",
                        _MAX_CUSTOM_RULES,
                    )
                    break

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
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                except (re.error, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping custom rule %r with invalid regex %r: %s",
                        name,
                        pattern,
                        exc,
                    )
                    continue

                self.custom_patterns.append(
                    {
                        "name": name[:_MAX_CUSTOM_NAME_LENGTH],
                        "pattern": pattern,
                        "compiled": compiled_pattern,
                        "reason": str(entry.get("reason", "manual_flag"))[
                            :_MAX_CUSTOM_REASON_LENGTH
                        ],
                        "description": str(entry.get("description", ""))[
                            :_MAX_CUSTOM_DESCRIPTION_LENGTH
                        ],
                    }
                )

                # Also register as a RoutingRule for get_rules() listing
                self.rules.append(
                    RoutingRule(
                        name=name[:_MAX_CUSTOM_NAME_LENGTH],
                        reason=str(entry.get("reason", "manual_flag"))[
                            :_MAX_CUSTOM_REASON_LENGTH
                        ],
                        description=str(
                            entry.get("description", f"Pattern: {pattern}")
                        )[:_MAX_CUSTOM_DESCRIPTION_LENGTH],
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
        decision = RoutingDecision()

        # Normalize validation_data: support both full report and quality sub-dict
        vd = validation_data or {}
        quality = vd.get("quality", vd)

        # Rule 1: Low confidence — only when overall_confidence is present.
        # Missing key must not default to 1.0 (S2-16 / Tier B residual).
        if self._rule_enabled("low_confidence") and vd and isinstance(quality, dict):
            if "overall_confidence" in quality:
                try:
                    confidence = float(quality["overall_confidence"])
                except (TypeError, ValueError):
                    confidence = 0.0  # unparseable → treat as low (fail closed)
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

        # Rule 5: Classification uncertain — only when confidence key present
        if self._rule_enabled("classification_uncertain") and classification_data:
            if "confidence" in classification_data:
                try:
                    cls_conf = float(classification_data["confidence"])
                except (TypeError, ValueError):
                    cls_conf = 0.0
                if cls_conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                    decision.should_route = True
                    decision.triggered_rules.append("classification_uncertain")
                    decision.reasons.append("classification_uncertain")
                    decision.metadata["classification_confidence"] = cls_conf

        # Rule 6: Custom patterns
        # Truncate text to guard against ReDoS on very large inputs.
        pattern_text = (
            extracted_text[:_MAX_PATTERN_TEXT_LENGTH] if extracted_text else ""
        )
        if pattern_text:
            for pattern_rule in self.custom_patterns:
                try:
                    compiled = pattern_rule.get("compiled")
                    if compiled is None:
                        compiled = re.compile(pattern_rule["pattern"], re.IGNORECASE)
                    if compiled.search(pattern_text):
                        decision.should_route = True
                        decision.triggered_rules.append(pattern_rule["name"])
                        decision.reasons.append(pattern_rule["reason"])
                        decision.metadata[f"pattern_{pattern_rule['name']}"] = True
                except re.error:
                    logger.warning(
                        "Regex error evaluating custom rule %r; skipping",
                        pattern_rule["name"],
                    )

        return decision

    def enqueue_decision(
        self,
        decision: RoutingDecision,
        *,
        job_id: str,
        review_queue: object,
        quality_classification: str = "",
    ) -> list[str]:
        """Enqueue review items for a positive routing decision.

        Post-sweep HIGH: ExceptionRouter previously only returned decisions;
        production HITL requires an explicit enqueue path into ReviewQueue.

        Args:
            decision: Result of :meth:`evaluate`.
            job_id: OCR job identifier.
            review_queue: Object with ``add(job_id, reason, confidence, ...)``.
            quality_classification: Optional quality label for the item.

        Returns:
            List of created review_id strings (empty when not routed).
        """
        if not decision.should_route:
            return []
        if not job_id or not str(job_id).strip():
            raise ValueError("job_id required to enqueue review items")
        if review_queue is None:
            raise ValueError("review_queue required when should_route is true (LT-13)")
        created: list[str] = []
        reasons = (
            list(decision.reasons)
            or list(decision.triggered_rules)
            or ["exception_routed"]
        )
        # One queue item per distinct reason (dedupe while preserving order).
        seen: set[str] = set()
        for reason in reasons:
            r = str(reason or "").strip() or "exception_routed"
            if r in seen:
                continue
            seen.add(r)
            extra_metadata = {
                "triggered_rules": list(decision.triggered_rules or []),
                "exception_router": True,
                "deliverable_hold": True,  # LT-13 HITL hold signal
                "api_job_id": str(job_id).strip(),
            }
            item = review_queue.add(  # type: ignore[attr-defined]
                job_id=str(job_id).strip(),
                reason=r,
                confidence=float(decision.confidence or 0.0),
                quality_classification=quality_classification
                or str(decision.metadata.get("quality_classification", "") or ""),
                metadata=dict(decision.metadata or {}) | extra_metadata,
            )
            rid = getattr(item, "review_id", None) or getattr(item, "id", None)
            if rid:
                created.append(str(rid))
        if not created:
            raise RuntimeError(
                "LT-13: exception routing should_route but no review items created"
            )
        return created

    def evaluate_and_enqueue(
        self,
        *,
        job_id: str,
        review_queue: object,
        validation_data: dict | None = None,
        classification_data: dict | None = None,
        handwriting_data: dict | None = None,
        extracted_text: str = "",
    ) -> tuple[RoutingDecision, list[str]]:
        """Evaluate then enqueue when ``should_route`` is true."""
        decision = self.evaluate(
            validation_data=validation_data,
            classification_data=classification_data,
            handwriting_data=handwriting_data,
            extracted_text=extracted_text,
        )
        vd = validation_data or {}
        quality = vd.get("quality", vd) if isinstance(vd, dict) else {}
        qclass = ""
        if isinstance(quality, dict):
            qclass = str(quality.get("classification", "") or "")
        ids = self.enqueue_decision(
            decision,
            job_id=job_id,
            review_queue=review_queue,
            quality_classification=qclass,
        )
        return decision, ids

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


# ---------------------------------------------------------------------------
# LT-13: deliverable hold markers (consumed by pipeline + download API)
# ---------------------------------------------------------------------------

HITL_HOLD_SUFFIX = ".hitl-hold.json"
_MAX_HOLD_WALK_ENTRIES = 10_000


def deliverable_hold_marker_path(output_pdf: str) -> str:
    """Return the sibling hold-marker path for a canonical PDF deliverable."""
    return f"{output_pdf}{HITL_HOLD_SUFFIX}"


def write_deliverable_hold_marker(
    output_pdf: str,
    *,
    decision: RoutingDecision | None = None,
    review_ids: list[str] | None = None,
    job_id: str = "",
    document_id: str = "",
) -> str:
    """Write a HITL hold marker next to *output_pdf* (LT-13).

    Presence of this marker means the deliverable must not be treated as
    released for compression/customer download until review clears it.
    Returns the marker path.
    """
    if not output_pdf or not str(output_pdf).strip():
        raise ValueError("output_pdf required for deliverable hold marker")
    path = deliverable_hold_marker_path(str(output_pdf))
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    payload = {
        "deliverable_hold": True,
        "output_pdf": str(output_pdf),
        "job_id": job_id or "",
        "document_id": document_id or "",
        "review_ids": list(review_ids or []),
        "should_route": bool(decision.should_route) if decision is not None else True,
        "triggered_rules": list(decision.triggered_rules)
        if decision is not None
        else [],
        "reasons": list(decision.reasons) if decision is not None else [],
        "confidence": float(decision.confidence) if decision is not None else 0.0,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    logger.warning(
        "LT-13 deliverable hold marker written: %s (reviews=%s)",
        path,
        payload["review_ids"],
    )
    return path


def is_deliverable_held(output_pdf: str) -> bool:
    """True when a HITL hold marker exists for *output_pdf* (LT-13).

    Checks:
    1. Sibling marker ``{path}.hitl-hold.json``
    2. If *path* is a job result directory, any ``EXPORT/**/*.hitl-hold.json``
    3. Walk parents for ``EXPORT/**/*.hitl-hold.json`` (sidecar download path)

    Raises OSError on unreadable trees so callers fail closed.
    """
    if not output_pdf:
        return False
    path = os.path.abspath(str(output_pdf))

    def _marker_exists(marker_path: str) -> bool:
        try:
            os.lstat(marker_path)
        except FileNotFoundError:
            return False
        except OSError:
            # A hold check is a release gate.  An unreadable marker path must
            # not silently turn into a downloadable artifact.
            raise
        # Presence is enough to hold the deliverable.  This also treats a
        # marker symlink conservatively instead of following it.
        return True

    if _marker_exists(deliverable_hold_marker_path(path)):
        return True

    def _export_has_hold(export_dir: str) -> bool:
        entries_seen = 0

        def _raise_walk_error(error: OSError) -> None:
            raise error

        for dirpath, dirnames, filenames in os.walk(
            export_dir,
            topdown=True,
            onerror=_raise_walk_error,
            followlinks=False,
        ):
            entries_seen += len(dirnames) + len(filenames)
            if entries_seen > _MAX_HOLD_WALK_ENTRIES:
                raise OSError("HITL hold tree exceeds the safety traversal limit")
            if any(os.path.islink(os.path.join(dirpath, name)) for name in dirnames):
                raise OSError("HITL hold tree contains a symlinked directory")
            for name in filenames:
                if name.endswith(HITL_HOLD_SUFFIX):
                    return True
        return False

    def _strict_is_dir(candidate: str) -> bool:
        """Check a directory without turning permission errors into false."""
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            return False
        except OSError:
            raise
        if stat.S_ISLNK(info.st_mode):
            raise OSError("HITL hold check encountered a symlinked directory")
        return stat.S_ISDIR(info.st_mode)

    path_is_dir = _strict_is_dir(path)
    if path_is_dir:
        export = os.path.join(path, "EXPORT")
        if _strict_is_dir(export) and _export_has_hold(export):
            return True
        # Also accept result_path that *is* EXPORT or contains holds
        if _export_has_hold(path):
            return True

    # Sidecar/other artifact: climb to find job EXPORT with a hold marker.
    root = path
    for _ in range(10):
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
        export = os.path.join(root, "EXPORT")
        if _strict_is_dir(export) and _export_has_hold(export):
            return True
    return False


def clear_deliverable_hold(output_pdf: str) -> bool:
    """Remove hold marker after human review release. Returns True if removed."""
    path = deliverable_hold_marker_path(str(output_pdf))
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def assert_deliverable_not_held(*paths: str) -> None:
    """Raise ValueError if any path is under HITL hold (LT-13 fail-closed).

    Callers that serve customer artifacts (jobs download, outputs API) must
    map ValueError → HTTP 423. OSError / ImportError must not be swallowed.
    """
    for p in paths:
        if not p:
            continue
        if is_deliverable_held(str(p)):
            raise ValueError(
                "Artifact is held for human review (LT-13 HITL). "
                "Clear the .hitl-hold.json marker after review release."
            )
