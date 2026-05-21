"""Entity relationship extraction for forensic OCR pipeline.

Detects relationships between consolidated entities (person->organization,
date->event, amount->purpose, etc.) using proximity heuristics, key-value
pair linkage, and regex-based patterns.

Designed to run *after* entity consolidation -- it reads the deduplicated
entity list and the raw page text to discover how entities relate to each
other within a document.

Relationship types detected:
- works_for:   PERSON -> ORG  (proximity + contextual patterns)
- located_at:  PERSON/ORG -> GPE/address  (proximity + contextual patterns)
- dated:       DATE -> any entity  (temporal proximity)
- amount_for:  amount -> purpose entity  (KV pair linkage + proximity)
- signed_by:   PERSON -> document  (signature-context patterns)
- references:  CASE_NUMBER/EXHIBIT_REF -> document  (legal reference patterns)
- sent_to:     PERSON -> PERSON  (correspondence patterns)
- sent_from:   PERSON -> PERSON  (correspondence patterns)

Output is appended to the consolidated .entities.json under a
``relationships`` key.

Graceful degradation: if the entity list is empty or contains fewer than
two entities, the extractor returns an empty list.  Relationship extraction
is opt-in and controlled by ``ENABLE_RELATIONSHIP_EXTRACTION``.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_RELATIONSHIP_EXTRACTION = os.environ.get(
    "ENABLE_RELATIONSHIP_EXTRACTION", "false"
).lower() in ("1", "true", "yes")

# Maximum character distance between two entity mentions for proximity-based
# relationship detection.  Tunable via environment variable.
_MAX_PROXIMITY_CHARS = int(os.environ.get("RELATIONSHIP_PROXIMITY_CHARS", "150"))

# Minimum confidence threshold for emitted relationships.
_MIN_CONFIDENCE = float(os.environ.get("RELATIONSHIP_MIN_CONFIDENCE", "0.4"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EntityRelationship:
    """A detected relationship between two entities."""

    source_id: str  # e.g. "ent_001"
    target_id: str  # e.g. "ent_003"
    relation_type: str  # e.g. "works_for", "dated", "amount_for"
    confidence: float = 0.0
    evidence: str = ""  # text snippet supporting the relationship
    page: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relation_type,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "page": self.page,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Relationship patterns (compiled once at module load)
# ---------------------------------------------------------------------------

# Patterns that indicate a PERSON works for an ORG.
_WORKS_FOR_PATTERNS = [
    re.compile(
        r"(?P<left>.{0,60}?)\bat\b(?P<right>.{0,60})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<left>.{0,60}?)\bof\b(?P<right>.{0,60})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<left>.{0,60}?)\bfrom\b(?P<right>.{0,60})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<left>.{0,60}?)\b(?:employed\s+by|works?\s+for|working\s+(?:for|at))\b(?P<right>.{0,60})",
        re.IGNORECASE,
    ),
]

# Patterns for "located_at" -- PERSON/ORG near a location/address.
_LOCATED_AT_PATTERNS = [
    re.compile(
        r"(?P<left>.{0,60}?)\b(?:located\s+(?:at|in)|based\s+in|headquartered\s+in|office\s+(?:at|in))\b(?P<right>.{0,60})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<left>.{0,60}?)\b(?:address|location)\s*[:]\s*(?P<right>.{0,80})",
        re.IGNORECASE,
    ),
]

# Patterns for "signed_by" -- signature context.
_SIGNED_BY_PATTERNS = [
    re.compile(
        r"\b(?:signed|executed|attested|witnessed)\s+by\b.{0,60}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsignature\s*[:]\s*.{0,60}",
        re.IGNORECASE,
    ),
]

# Patterns for correspondence (sent_to / sent_from).
_CORRESPONDENCE_TO_PATTERNS = [
    re.compile(r"\bTo\s*[:]\s*(?P<target>.{0,80})", re.IGNORECASE),
    re.compile(r"\bDear\s+(?P<target>.{0,60})", re.IGNORECASE),
    re.compile(r"\bAttn\s*[:.]?\s*(?P<target>.{0,60})", re.IGNORECASE),
]

_CORRESPONDENCE_FROM_PATTERNS = [
    re.compile(r"\bFrom\s*[:]\s*(?P<source>.{0,80})", re.IGNORECASE),
    re.compile(r"\bSender\s*[:]\s*(?P<source>.{0,80})", re.IGNORECASE),
]

# Patterns for legal references.
_REFERENCE_PATTERNS = [
    re.compile(
        r"\b(?:re|regarding|in\s+re|in\s+the\s+matter\s+of)\s*[:]\s*.{0,80}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:see|refer\s+to|see\s+also)\s+(?:Exhibit|Ex\.?)\b.{0,40}",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Entity type normalization helpers
# ---------------------------------------------------------------------------

# Map entity types from various sources to canonical categories.
_PERSON_TYPES = frozenset({"person", "person_name"})
_ORG_TYPES = frozenset({"org", "organization"})
_LOCATION_TYPES = frozenset({"gpe", "address", "location"})
_DATE_TYPES = frozenset({"date"})
_AMOUNT_TYPES = frozenset({"amount", "money"})
_CASE_TYPES = frozenset({"case_number"})
_EXHIBIT_TYPES = frozenset({"exhibit_ref"})
_REFERENCE_TYPES = frozenset({"reference_number"})


def _canonical_type(entity_type: str) -> str:
    """Return a canonical category for the given entity type."""
    t = entity_type.strip().lower()
    if t in _PERSON_TYPES:
        return "person"
    if t in _ORG_TYPES:
        return "org"
    if t in _LOCATION_TYPES:
        return "location"
    if t in _DATE_TYPES:
        return "date"
    if t in _AMOUNT_TYPES:
        return "amount"
    if t in _CASE_TYPES:
        return "case_number"
    if t in _EXHIBIT_TYPES:
        return "exhibit_ref"
    if t in _REFERENCE_TYPES:
        return "reference_number"
    return t


def _is_person(entity: dict) -> bool:
    return _canonical_type(entity.get("type", "")) == "person"


def _is_org(entity: dict) -> bool:
    return _canonical_type(entity.get("type", "")) == "org"


def _is_location(entity: dict) -> bool:
    return _canonical_type(entity.get("type", "")) == "location"


def _is_date(entity: dict) -> bool:
    return _canonical_type(entity.get("type", "")) == "date"


def _is_amount(entity: dict) -> bool:
    return _canonical_type(entity.get("type", "")) == "amount"


def _is_case_or_exhibit(entity: dict) -> bool:
    ct = _canonical_type(entity.get("type", ""))
    return ct in ("case_number", "exhibit_ref", "reference_number")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _entity_offsets(entity: dict) -> tuple:
    """Return (start, end) character offsets from entity metadata.

    Entities may carry offsets directly or inside ``metadata``.
    """
    meta = entity.get("metadata", {})
    start = meta.get("start", entity.get("start", -1))
    end = meta.get("end", entity.get("end", -1))
    return int(start), int(end)


def _extract_evidence(text: str, start: int, end: int, context: int = 40) -> str:
    """Extract a snippet of *text* around [start, end] with *context* chars."""
    if not text:
        return ""
    lo = max(0, start - context)
    hi = min(len(text), end + context)
    snippet = text[lo:hi].strip()
    # Collapse whitespace
    snippet = re.sub(r"\s+", " ", snippet)
    # Cap length
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."
    return snippet


def _char_distance(ent_a: dict, ent_b: dict) -> int:
    """Return the minimum character distance between two entity spans.

    Returns a large sentinel if offsets are unavailable.
    """
    a_start, a_end = _entity_offsets(ent_a)
    b_start, b_end = _entity_offsets(ent_b)

    if a_start < 0 or b_start < 0:
        return 999999

    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    # Overlapping
    return 0


def _entity_in_span(entity: dict, text: str, span_start: int, span_end: int) -> bool:
    """Check whether the entity text appears inside text[span_start:span_end]."""
    ent_text = entity.get("text", "")
    if not ent_text:
        return False

    e_start, e_end = _entity_offsets(entity)
    # If we have valid offsets, check overlap
    if e_start >= 0 and e_end > 0:
        return e_start >= span_start and e_end <= span_end

    # Fallback: substring search within the span
    span_text = text[span_start:span_end] if text else ""
    return ent_text.lower() in span_text.lower()


# ---------------------------------------------------------------------------
# Proximity-based relationship detection
# ---------------------------------------------------------------------------


def _proximity_relationships(
    entities: list,
    page_texts: dict,
    max_distance: int,
) -> list:
    """Detect relationships based on entity proximity in text.

    Two entities on the same page within *max_distance* characters are
    candidates.  The relationship type is inferred from their canonical
    types:
    - person + org   -> works_for
    - person/org + location -> located_at
    - date + any     -> dated
    - amount + any   -> amount_for

    Args:
        entities: Deduplicated entity dicts from consolidation.
        page_texts: Mapping of page number -> full page text.
        max_distance: Maximum character gap for proximity.

    Returns:
        List of EntityRelationship instances.
    """
    relationships = []
    n = len(entities)

    for i in range(n):
        for j in range(i + 1, n):
            ent_a = entities[i]
            ent_b = entities[j]

            # Must be on the same page
            if ent_a.get("page", 0) != ent_b.get("page", 0):
                continue

            dist = _char_distance(ent_a, ent_b)
            if dist > max_distance:
                continue

            page = ent_a.get("page", 0)
            text = page_texts.get(page, "")

            # Compute confidence: closer entities get higher confidence
            if max_distance > 0:
                proximity_score = max(0.0, 1.0 - (dist / max_distance))
            else:
                proximity_score = 1.0

            # Determine relationship type based on entity type pair
            rel = _infer_proximity_relation(ent_a, ent_b, proximity_score, text)
            if rel is not None:
                relationships.append(rel)

    return relationships


def _infer_proximity_relation(
    ent_a: dict,
    ent_b: dict,
    proximity_score: float,
    text: str,
) -> Optional[EntityRelationship]:
    """Infer a relationship from two co-located entities.

    Returns an EntityRelationship or None if no valid relationship.
    """
    ca = _canonical_type(ent_a.get("type", ""))
    cb = _canonical_type(ent_b.get("type", ""))
    page = ent_a.get("page", 0)

    a_start, a_end = _entity_offsets(ent_a)
    b_start, b_end = _entity_offsets(ent_b)
    ev_start = min(a_start, b_start) if a_start >= 0 and b_start >= 0 else 0
    ev_end = max(a_end, b_end) if a_end > 0 and b_end > 0 else 0
    evidence = _extract_evidence(text, ev_start, ev_end)

    # person + org -> works_for
    if ca == "person" and cb == "org":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="works_for",
            confidence=round(proximity_score * 0.6, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )
    if ca == "org" and cb == "person":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="works_for",
            confidence=round(proximity_score * 0.6, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )

    # person/org + location -> located_at
    if ca in ("person", "org") and cb == "location":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="located_at",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )
    if cb in ("person", "org") and ca == "location":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="located_at",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )

    # date + any non-date -> dated
    if ca == "date" and cb != "date":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="dated",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )
    if cb == "date" and ca != "date":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="dated",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )

    # amount + any non-amount -> amount_for
    if ca == "amount" and cb != "amount":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="amount_for",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )
    if cb == "amount" and ca != "amount":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="amount_for",
            confidence=round(proximity_score * 0.5, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "proximity"},
        )

    return None


# ---------------------------------------------------------------------------
# KV pair linkage
# ---------------------------------------------------------------------------


def _kv_linkage_relationships(
    entities: list,
    kv_pairs: list,
) -> list:
    """Detect relationships from key-value pair structure.

    When a KV pair's value matches an entity text and the key matches
    another entity type, that implies a relationship between the two
    entities.

    For example, if kv_pair key="person_name" value="John Smith" and
    there is an entity "Acme Corp" (ORG) on the same page, and
    another kv_pair key="organization" value="Acme Corp", then
    John Smith works_for Acme Corp.

    Args:
        entities: Deduplicated entity dicts.
        kv_pairs: Key-value pair dicts from consolidation.

    Returns:
        List of EntityRelationship instances.
    """
    if not entities or not kv_pairs:
        return []

    relationships = []

    # Build a lookup: (normalized_text, page) -> entity
    entity_lookup = {}
    for ent in entities:
        key = (ent.get("text", "").strip().lower(), ent.get("page", 0))
        entity_lookup[key] = ent

    # Group KV pairs by page
    page_kv_groups = {}
    for kv in kv_pairs:
        pg = kv.get("page", 0)
        if pg not in page_kv_groups:
            page_kv_groups[pg] = []
        page_kv_groups[pg].append(kv)

    # For each page, look for pairwise KV linkages
    for pg, kvs in page_kv_groups.items():
        # Find pairs of KV entries that link to different entities
        for i, kv_a in enumerate(kvs):
            for j in range(i + 1, len(kvs)):
                kv_b = kvs[j]

                ent_a = entity_lookup.get(
                    (kv_a.get("value", "").strip().lower(), pg)
                )
                ent_b = entity_lookup.get(
                    (kv_b.get("value", "").strip().lower(), pg)
                )

                if ent_a is None or ent_b is None:
                    continue
                if ent_a["id"] == ent_b["id"]:
                    continue

                rel = _infer_kv_relation(ent_a, ent_b, kv_a, kv_b, pg)
                if rel is not None:
                    relationships.append(rel)

    return relationships


def _infer_kv_relation(
    ent_a: dict,
    ent_b: dict,
    kv_a: dict,
    kv_b: dict,
    page: int,
) -> Optional[EntityRelationship]:
    """Infer a relationship between two entities linked via KV pairs."""
    ca = _canonical_type(ent_a.get("type", ""))
    cb = _canonical_type(ent_b.get("type", ""))

    avg_conf = (
        kv_a.get("confidence", 0.0) + kv_b.get("confidence", 0.0)
    ) / 2.0

    evidence = f"{kv_a.get('key', '')}: {kv_a.get('value', '')} | {kv_b.get('key', '')}: {kv_b.get('value', '')}"

    # person + org -> works_for
    if ca == "person" and cb == "org":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="works_for",
            confidence=round(avg_conf * 0.8, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )
    if ca == "org" and cb == "person":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="works_for",
            confidence=round(avg_conf * 0.8, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )

    # person/org + location -> located_at
    if ca in ("person", "org") and cb == "location":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="located_at",
            confidence=round(avg_conf * 0.7, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )
    if cb in ("person", "org") and ca == "location":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="located_at",
            confidence=round(avg_conf * 0.7, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )

    # amount + anything -> amount_for
    if ca == "amount" and cb != "amount":
        return EntityRelationship(
            source_id=ent_a["id"],
            target_id=ent_b["id"],
            relation_type="amount_for",
            confidence=round(avg_conf * 0.7, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )
    if cb == "amount" and ca != "amount":
        return EntityRelationship(
            source_id=ent_b["id"],
            target_id=ent_a["id"],
            relation_type="amount_for",
            confidence=round(avg_conf * 0.7, 4),
            evidence=evidence,
            page=page,
            metadata={"method": "kv_linkage"},
        )

    return None


# ---------------------------------------------------------------------------
# Pattern-based relationship detection
# ---------------------------------------------------------------------------


def _pattern_relationships(
    entities: list,
    page_texts: dict,
) -> list:
    """Detect relationships using regex patterns in page text.

    Scans each page for patterns indicating specific relationship types
    (works_for, signed_by, sent_to, sent_from, references) and checks
    whether known entities appear within the matched spans.

    Args:
        entities: Deduplicated entity dicts.
        page_texts: Mapping of page number -> full page text.

    Returns:
        List of EntityRelationship instances.
    """
    relationships = []

    # Group entities by page for efficient lookup
    page_entities = {}
    for ent in entities:
        pg = ent.get("page", 0)
        if pg not in page_entities:
            page_entities[pg] = []
        page_entities[pg].append(ent)

    for pg, text in page_texts.items():
        if not text:
            continue

        ents = page_entities.get(pg, [])
        if not ents:
            continue

        # --- works_for patterns ---
        relationships.extend(
            _scan_pattern_pair(
                text, ents, pg,
                _WORKS_FOR_PATTERNS,
                source_filter=_is_person,
                target_filter=_is_org,
                relation_type="works_for",
                base_confidence=0.75,
            )
        )

        # --- located_at patterns ---
        relationships.extend(
            _scan_pattern_pair(
                text, ents, pg,
                _LOCATED_AT_PATTERNS,
                source_filter=lambda e: _is_person(e) or _is_org(e),
                target_filter=_is_location,
                relation_type="located_at",
                base_confidence=0.70,
            )
        )

        # --- signed_by patterns ---
        for pat in _SIGNED_BY_PATTERNS:
            for m in pat.finditer(text):
                span_start = m.start()
                span_end = m.end()
                persons = [
                    e for e in ents
                    if _is_person(e)
                    and _entity_in_span(e, text, span_start, span_end)
                ]
                for person in persons:
                    relationships.append(EntityRelationship(
                        source_id=person["id"],
                        target_id="document",
                        relation_type="signed_by",
                        confidence=0.70,
                        evidence=_extract_evidence(text, span_start, span_end),
                        page=pg,
                        metadata={"method": "pattern"},
                    ))

        # --- sent_to patterns ---
        for pat in _CORRESPONDENCE_TO_PATTERNS:
            for m in pat.finditer(text):
                span_start = m.start()
                span_end = m.end()
                persons = [
                    e for e in ents
                    if _is_person(e)
                    and _entity_in_span(e, text, span_start, span_end)
                ]
                for person in persons:
                    # Find a "from" person on the same page
                    from_persons = [
                        e for e in ents
                        if _is_person(e) and e["id"] != person["id"]
                    ]
                    if from_persons:
                        relationships.append(EntityRelationship(
                            source_id=from_persons[0]["id"],
                            target_id=person["id"],
                            relation_type="sent_to",
                            confidence=0.65,
                            evidence=_extract_evidence(text, span_start, span_end),
                            page=pg,
                            metadata={"method": "pattern"},
                        ))
                    else:
                        # No "from" person found, still record the addressee
                        relationships.append(EntityRelationship(
                            source_id="document",
                            target_id=person["id"],
                            relation_type="sent_to",
                            confidence=0.60,
                            evidence=_extract_evidence(text, span_start, span_end),
                            page=pg,
                            metadata={"method": "pattern"},
                        ))

        # --- sent_from patterns ---
        for pat in _CORRESPONDENCE_FROM_PATTERNS:
            for m in pat.finditer(text):
                span_start = m.start()
                span_end = m.end()
                persons = [
                    e for e in ents
                    if _is_person(e)
                    and _entity_in_span(e, text, span_start, span_end)
                ]
                for person in persons:
                    relationships.append(EntityRelationship(
                        source_id=person["id"],
                        target_id="document",
                        relation_type="sent_from",
                        confidence=0.65,
                        evidence=_extract_evidence(text, span_start, span_end),
                        page=pg,
                        metadata={"method": "pattern"},
                    ))

        # --- references patterns ---
        for pat in _REFERENCE_PATTERNS:
            for m in pat.finditer(text):
                span_start = m.start()
                span_end = m.end()
                refs = [
                    e for e in ents
                    if _is_case_or_exhibit(e)
                    and _entity_in_span(e, text, span_start, span_end)
                ]
                for ref_ent in refs:
                    relationships.append(EntityRelationship(
                        source_id=ref_ent["id"],
                        target_id="document",
                        relation_type="references",
                        confidence=0.80,
                        evidence=_extract_evidence(text, span_start, span_end),
                        page=pg,
                        metadata={"method": "pattern"},
                    ))

    return relationships


def _scan_pattern_pair(
    text: str,
    entities: list,
    page: int,
    patterns: list,
    source_filter,
    target_filter,
    relation_type: str,
    base_confidence: float,
) -> list:
    """Scan text for patterns and look for source/target entity pairs.

    For patterns with named groups ``left`` and ``right``, entities matching
    the source filter in the left span and the target filter in the right
    span (or vice-versa) form a relationship.

    Returns:
        List of EntityRelationship instances.
    """
    rels = []

    for pat in patterns:
        for m in pat.finditer(text):
            span_start = m.start()
            span_end = m.end()

            # Try named groups first (left/right of the keyword)
            try:
                left_start = m.start("left")
                left_end = m.end("left")
                right_start = m.start("right")
                right_end = m.end("right")
            except IndexError:
                # No named groups -- use full match span
                left_start = span_start
                left_end = span_end
                right_start = span_start
                right_end = span_end

            # Find source entities in left span, target in right span
            sources = [
                e for e in entities
                if source_filter(e)
                and _entity_in_span(e, text, left_start, left_end)
            ]
            targets = [
                e for e in entities
                if target_filter(e)
                and _entity_in_span(e, text, right_start, right_end)
            ]

            for src in sources:
                for tgt in targets:
                    if src["id"] == tgt["id"]:
                        continue
                    rels.append(EntityRelationship(
                        source_id=src["id"],
                        target_id=tgt["id"],
                        relation_type=relation_type,
                        confidence=base_confidence,
                        evidence=_extract_evidence(text, span_start, span_end),
                        page=page,
                        metadata={"method": "pattern"},
                    ))

            # Also try reversed: source in right, target in left
            sources_r = [
                e for e in entities
                if source_filter(e)
                and _entity_in_span(e, text, right_start, right_end)
            ]
            targets_r = [
                e for e in entities
                if target_filter(e)
                and _entity_in_span(e, text, left_start, left_end)
            ]

            for src in sources_r:
                for tgt in targets_r:
                    if src["id"] == tgt["id"]:
                        continue
                    rels.append(EntityRelationship(
                        source_id=src["id"],
                        target_id=tgt["id"],
                        relation_type=relation_type,
                        confidence=base_confidence,
                        evidence=_extract_evidence(text, span_start, span_end),
                        page=page,
                        metadata={"method": "pattern"},
                    ))

    return rels


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_relationships(relationships: list) -> list:
    """Remove duplicate relationships, keeping the highest-confidence one.

    Two relationships are considered duplicates if they share the same
    (source_id, target_id, relation_type).

    Args:
        relationships: List of EntityRelationship instances.

    Returns:
        Deduplicated list sorted by confidence descending.
    """
    if not relationships:
        return []

    best = {}
    for rel in relationships:
        key = (rel.source_id, rel.target_id, rel.relation_type)
        if key not in best or rel.confidence > best[key].confidence:
            best[key] = rel

    result = sorted(best.values(), key=lambda r: (-r.confidence, r.source_id))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RelationshipExtractor:
    """Extract relationships between consolidated entities.

    Combines three detection strategies:
    1. **Proximity**: Entities near each other on the same page.
    2. **KV linkage**: Key-value pairs that connect entities.
    3. **Pattern**: Regex patterns indicating specific relationships.

    Usage::

        extractor = RelationshipExtractor()
        rels = extractor.extract_relationships(
            entities=consolidated["entities"],
            page_texts={1: "John Smith at Acme Corp..."},
            kv_pairs=consolidated.get("key_value_pairs", []),
        )
    """

    def __init__(
        self,
        max_proximity_chars: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ):
        self._max_proximity = (
            max_proximity_chars
            if max_proximity_chars is not None
            else _MAX_PROXIMITY_CHARS
        )
        self._min_confidence = (
            min_confidence
            if min_confidence is not None
            else _MIN_CONFIDENCE
        )

    def extract_relationships(
        self,
        entities: list,
        page_texts: Optional[dict] = None,
        kv_pairs: Optional[list] = None,
    ) -> list:
        """Detect relationships between entities.

        Args:
            entities: Deduplicated entity dicts (from consolidate_entities).
            page_texts: Optional mapping of page number -> full page text.
                        Required for proximity and pattern detection.
            kv_pairs: Optional key-value pair dicts (from consolidation).

        Returns:
            List of EntityRelationship instances, deduplicated and filtered
            by minimum confidence.
        """
        if not entities or len(entities) < 2:
            return []

        all_rels = []

        if page_texts:
            all_rels.extend(
                _proximity_relationships(entities, page_texts, self._max_proximity)
            )
            all_rels.extend(
                _pattern_relationships(entities, page_texts)
            )

        if kv_pairs:
            all_rels.extend(
                _kv_linkage_relationships(entities, kv_pairs)
            )

        # Deduplicate and filter
        deduped = deduplicate_relationships(all_rels)
        filtered = [r for r in deduped if r.confidence >= self._min_confidence]

        return filtered


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


def extract_and_attach_relationships(
    consolidated: dict,
    page_texts: Optional[dict] = None,
    max_proximity_chars: Optional[int] = None,
    min_confidence: Optional[float] = None,
) -> dict:
    """Extract relationships and attach them to a consolidated entity dict.

    This is the primary integration point.  Call this after
    ``consolidate_entities()`` to enrich the output with relationship data.

    Args:
        consolidated: Dict from ``consolidate_entities()``.
        page_texts: Mapping of page number -> full page text.
        max_proximity_chars: Override for proximity distance threshold.
        min_confidence: Override for minimum confidence filter.

    Returns:
        The same dict with a ``relationships`` key added and the summary
        updated with ``total_relationships``.
    """
    entities = consolidated.get("entities", [])
    kv_pairs = consolidated.get("key_value_pairs", [])

    if len(entities) < 2:
        consolidated["relationships"] = []
        consolidated["summary"]["total_relationships"] = 0
        return consolidated

    extractor = RelationshipExtractor(
        max_proximity_chars=max_proximity_chars,
        min_confidence=min_confidence,
    )

    rels = extractor.extract_relationships(
        entities=entities,
        page_texts=page_texts,
        kv_pairs=kv_pairs,
    )

    consolidated["relationships"] = [r.to_dict() for r in rels]
    consolidated["summary"]["total_relationships"] = len(rels)

    # Add relationship type breakdown to summary
    rel_type_counts = {}
    for r in rels:
        rel_type_counts[r.relation_type] = rel_type_counts.get(r.relation_type, 0) + 1
    consolidated["summary"]["relationship_types"] = dict(sorted(rel_type_counts.items()))

    return consolidated
