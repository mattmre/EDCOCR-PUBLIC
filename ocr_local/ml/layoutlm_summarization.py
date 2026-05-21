"""CTC-safe extractive summarization. Selects existing sentences — never generates text.

Document-level semantic summarization for the forensic OCR pipeline.
All methods are **extractive only**: they rank and select existing
sentences from the source document.  No generative AI, no paraphrasing,
no text rewriting — this is required for CTC (zero hallucination)
compliance.

Scoring methods:
- TextRank: Graph-based sentence ranking using word overlap similarity.
- Entity Density: Scores sentences by count of recognized entities.
- Layout Position: Scores by document position (titles, headers, first/last
  paragraphs get higher scores).
- Combined: Weighted combination of all three.

Pure Python — no ML imports (no torch, transformers, nltk, spacy).

Environment Variables:
    LAYOUTLM_SUMMARIZATION_METHOD (str):
        Summarization method: textrank, entity_density, layout_position,
        or combined.  Default: ``combined``.
    LAYOUTLM_SUMMARY_MAX_SENTENCES (int):
        Maximum number of sentences to include in the summary.
        Default: ``5``.
"""

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

_DEFAULT_METHOD = os.environ.get(
    "LAYOUTLM_SUMMARIZATION_METHOD", "combined"
).lower().strip()

_DEFAULT_MAX_SENTENCES = max(
    1,
    int(os.environ.get("LAYOUTLM_SUMMARY_MAX_SENTENCES", "5")),
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SummarizationMethod(Enum):
    """Available extractive summarization scoring methods."""

    TEXTRANK = "textrank"
    ENTITY_DENSITY = "entity_density"
    LAYOUT_POSITION = "layout_position"
    COMBINED = "combined"


# Map env-var string values to enum members
_METHOD_LOOKUP = {m.value: m for m in SummarizationMethod}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SummarizationConfig:
    """Configuration for document summarization."""

    method: SummarizationMethod = SummarizationMethod.COMBINED
    max_sentences: int = 5
    min_sentence_length: int = 10
    entity_weight: float = 0.4
    position_weight: float = 0.3
    textrank_weight: float = 0.3
    include_metadata: bool = True

    def __post_init__(self):
        # Clamp weights to [0, 1]
        self.entity_weight = max(0.0, min(1.0, float(self.entity_weight)))
        self.position_weight = max(0.0, min(1.0, float(self.position_weight)))
        self.textrank_weight = max(0.0, min(1.0, float(self.textrank_weight)))
        self.max_sentences = max(1, int(self.max_sentences))
        self.min_sentence_length = max(0, int(self.min_sentence_length))


@dataclass
class SummarySentence:
    """A single sentence selected for the summary."""

    text: str
    page_num: int = 0
    score: float = 0.0
    method: str = ""
    bbox: Optional[list] = None
    position_in_page: float = 0.0


@dataclass
class DocumentSummary:
    """Complete extractive summary of a document."""

    sentences: list = field(default_factory=list)
    document_id: str = ""
    total_pages: int = 0
    total_sentences: int = 0
    entity_summary: dict = field(default_factory=dict)
    config_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default config from environment
# ---------------------------------------------------------------------------


def _default_config() -> SummarizationConfig:
    """Build a SummarizationConfig from environment variables."""
    method = _METHOD_LOOKUP.get(_DEFAULT_METHOD, SummarizationMethod.COMBINED)
    return SummarizationConfig(
        method=method,
        max_sentences=_DEFAULT_MAX_SENTENCES,
    )


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list:
    """Split text into sentences using regex on sentence-ending punctuation.

    Splits on ``.``, ``!``, or ``?`` followed by whitespace or end-of-string.
    Preserves the terminating punctuation with the sentence.

    Args:
        text: Input text string.

    Returns:
        List of sentence strings (stripped, non-empty).
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by space/newline/end
    parts = re.split(r'(?<=[.!?])(?:\s+|\Z)', text.strip())
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# TextRank scoring (pure Python — no networkx)
# ---------------------------------------------------------------------------


def _word_set(sentence: str) -> set:
    """Extract a set of lowercased words from a sentence."""
    return set(re.findall(r'[a-zA-Z0-9]+', sentence.lower()))


def _sentence_similarity(s1: str, s2: str) -> float:
    """Compute word-overlap similarity between two sentences.

    Returns the Jaccard-like overlap: |intersection| / log(|s1| + |s2|).
    Uses log normalization to avoid biasing toward long sentences.
    """
    w1 = _word_set(s1)
    w2 = _word_set(s2)

    if not w1 or not w2:
        return 0.0

    overlap = len(w1 & w2)
    if overlap == 0:
        return 0.0

    # Log-normalized overlap to reduce length bias
    denominator = math.log(len(w1) + len(w2))
    if denominator == 0.0:
        return 0.0

    return float(overlap) / denominator


def _textrank_scores(
    sentences: list,
    damping: float = 0.85,
    iterations: int = 30,
) -> list:
    """Compute TextRank scores for a list of sentences.

    Implements the TextRank algorithm using an adjacency matrix built from
    word-overlap similarity.  Pure Python — no external graph libraries.

    Args:
        sentences: List of sentence strings.
        damping: Damping factor (probability of following a link). Default 0.85.
        iterations: Number of power-iteration steps. Default 30.

    Returns:
        List of float scores (same length as sentences), normalized to [0, 1].
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Build adjacency matrix
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _sentence_similarity(sentences[i], sentences[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    # Compute outgoing weight sums for normalization
    out_sums = [sum(matrix[i]) for i in range(n)]

    # Initialize scores uniformly
    scores = [1.0 / n] * n

    # Power iteration
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if i != j and out_sums[j] > 0.0:
                    rank_sum += (matrix[j][i] / out_sums[j]) * scores[j]
            new_scores[i] = (1.0 - damping) / n + damping * rank_sum
        scores = new_scores

    # Normalize to [0, 1]
    max_score = max(scores) if scores else 1.0
    if max_score > 0.0:
        scores = [s / max_score for s in scores]

    return scores


# ---------------------------------------------------------------------------
# Entity density scoring
# ---------------------------------------------------------------------------


def _entity_density_scores(
    sentences: list,
    entities: list,
) -> list:
    """Score sentences by the density of recognized entities they contain.

    For each sentence, counts how many entity texts appear in it (case-
    insensitive substring match).  Scores are normalized to [0, 1].

    Args:
        sentences: List of sentence strings.
        entities: List of entity dicts, each with at least a ``text`` key.

    Returns:
        List of float scores (same length as sentences), normalized to [0, 1].
    """
    n = len(sentences)
    if n == 0:
        return []

    if not entities:
        return [0.0] * n

    # Extract entity texts
    entity_texts = []
    for ent in entities:
        if isinstance(ent, dict):
            text = ent.get("text", "")
        else:
            text = str(getattr(ent, "text", ""))
        if text.strip():
            entity_texts.append(text.strip().lower())

    if not entity_texts:
        return [0.0] * n

    counts = []
    for sent in sentences:
        sent_lower = sent.lower()
        count = sum(1 for et in entity_texts if et in sent_lower)
        counts.append(float(count))

    max_count = max(counts) if counts else 1.0
    if max_count > 0.0:
        return [c / max_count for c in counts]
    return [0.0] * n


# ---------------------------------------------------------------------------
# Layout position scoring
# ---------------------------------------------------------------------------


def _layout_position_scores(
    sentences: list,
    page_indices: list,
    total_pages: int,
    layout_regions: list,
) -> list:
    """Score sentences by their position in the document layout.

    Heuristics applied:
    - First page gets a bonus.
    - First and last sentences on each page get a bonus.
    - Sentences matching layout region text of type title/header get a bonus.
    - Position within page: earlier sentences score slightly higher.

    Args:
        sentences: List of sentence strings.
        page_indices: List of (page_number, position_in_page) tuples for each
            sentence, where position_in_page is a float in [0, 1].
        total_pages: Total number of pages in the document.
        layout_regions: List of layout region dicts with ``type`` and ``text``.

    Returns:
        List of float scores (same length as sentences), normalized to [0, 1].
    """
    n = len(sentences)
    if n == 0:
        return []

    # Collect title/header texts for matching
    header_texts = set()
    if layout_regions:
        for region in layout_regions:
            if isinstance(region, dict):
                rtype = region.get("type", "").lower()
                rtext = region.get("text", "").strip().lower()
            else:
                rtype = str(getattr(region, "type", "")).lower()
                rtext = str(getattr(region, "text", "")).strip().lower()
            if rtype in ("title", "header", "heading") and rtext:
                header_texts.add(rtext)

    raw_scores = []
    for idx, sent in enumerate(sentences):
        score = 0.0
        page_num, pos_in_page = page_indices[idx] if idx < len(page_indices) else (0, 0.5)

        # First page bonus (0-indexed)
        if page_num == 0:
            score += 0.3

        # Last page bonus (smaller)
        if total_pages > 1 and page_num == total_pages - 1:
            score += 0.1

        # Position within page: earlier = higher score
        # pos_in_page ∈ [0, 1]; invert so 0 → 0.2, 1 → 0.0
        score += 0.2 * (1.0 - pos_in_page)

        # First/last sentence overall bonus
        if idx == 0:
            score += 0.2
        if idx == n - 1:
            score += 0.1

        # Header/title match bonus
        sent_lower = sent.strip().lower()
        for ht in header_texts:
            if ht in sent_lower or sent_lower in ht:
                score += 0.3
                break

        raw_scores.append(score)

    # Normalize to [0, 1]
    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score > 0.0:
        return [s / max_score for s in raw_scores]
    return [0.0] * n


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------


def _combined_scores(
    sentences: list,
    entities: list,
    page_indices: list,
    total_pages: int,
    layout_regions: list,
    config: SummarizationConfig,
) -> list:
    """Compute weighted combination of all three scoring methods.

    Args:
        sentences: List of sentence strings.
        entities: Entity dicts for entity-density scoring.
        page_indices: (page_num, position_in_page) per sentence.
        total_pages: Total document pages.
        layout_regions: Layout region dicts.
        config: Summarization config with weight values.

    Returns:
        List of combined scores, normalized to [0, 1].
    """
    n = len(sentences)
    if n == 0:
        return []

    tr_scores = _textrank_scores(sentences)
    ed_scores = _entity_density_scores(sentences, entities)
    lp_scores = _layout_position_scores(
        sentences, page_indices, total_pages, layout_regions,
    )

    combined = []
    for i in range(n):
        score = (
            config.textrank_weight * tr_scores[i]
            + config.entity_weight * ed_scores[i]
            + config.position_weight * lp_scores[i]
        )
        combined.append(score)

    max_score = max(combined) if combined else 1.0
    if max_score > 0.0:
        return [s / max_score for s in combined]
    return [0.0] * n


# ---------------------------------------------------------------------------
# Entity summary helper
# ---------------------------------------------------------------------------


def _build_entity_summary(entities: list) -> dict:
    """Build entity type → count mapping from entity dicts.

    Args:
        entities: List of entity dicts with ``type`` or ``label`` key.

    Returns:
        Dict mapping entity type strings to occurrence counts.
    """
    counts = {}
    if not entities:
        return counts

    for ent in entities:
        if isinstance(ent, dict):
            etype = ent.get("type", ent.get("label", "UNKNOWN"))
        else:
            etype = getattr(ent, "type", getattr(ent, "label", "UNKNOWN"))
        etype = str(etype) if etype else "UNKNOWN"
        counts[etype] = counts.get(etype, 0) + 1

    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Main summarization function
# ---------------------------------------------------------------------------


def summarize_document(
    pages_text: list,
    entities: Optional[list] = None,
    layout_regions: Optional[list] = None,
    config: Optional[SummarizationConfig] = None,
    document_id: str = "",
) -> DocumentSummary:
    """Produce an extractive summary of a multi-page document.

    Splits page texts into sentences, scores them using the configured
    method, and returns the top-N sentences.  **Extractive only** — all
    returned sentences are verbatim copies from the source text.

    Args:
        pages_text: List of per-page text strings.
        entities: Optional list of entity dicts (from .entities.json).
            Each dict should have at least a ``text`` key.
        layout_regions: Optional list of layout region dicts (from
            structure.json).  Each dict should have ``type`` and ``text``.
        config: Summarization configuration.  If None, uses defaults
            (from environment variables).
        document_id: Optional document identifier for metadata.

    Returns:
        DocumentSummary with selected sentences and metadata.
    """
    if config is None:
        config = _default_config()
    if entities is None:
        entities = []
    if layout_regions is None:
        layout_regions = []

    total_pages = len(pages_text)

    # Split all pages into sentences with page tracking
    all_sentences = []
    page_indices = []  # (page_num, position_in_page)

    for page_num, page_text in enumerate(pages_text):
        page_sents = _split_sentences(page_text)
        for sent_idx, sent in enumerate(page_sents):
            # Filter by minimum length
            if len(sent) < config.min_sentence_length:
                continue
            all_sentences.append(sent)
            pos = sent_idx / max(len(page_sents), 1)
            page_indices.append((page_num, pos))

    total_sentences = len(all_sentences)

    if total_sentences == 0:
        return DocumentSummary(
            sentences=[],
            document_id=document_id,
            total_pages=total_pages,
            total_sentences=0,
            entity_summary=_build_entity_summary(entities),
            config_used=_config_to_dict(config),
        )

    # Score sentences based on method
    method = config.method
    method_label = method.value

    if method == SummarizationMethod.TEXTRANK:
        scores = _textrank_scores(all_sentences)
    elif method == SummarizationMethod.ENTITY_DENSITY:
        scores = _entity_density_scores(all_sentences, entities)
    elif method == SummarizationMethod.LAYOUT_POSITION:
        scores = _layout_position_scores(
            all_sentences, page_indices, total_pages, layout_regions,
        )
    elif method == SummarizationMethod.COMBINED:
        scores = _combined_scores(
            all_sentences, entities, page_indices, total_pages,
            layout_regions, config,
        )
    else:
        # Fallback to combined
        scores = _combined_scores(
            all_sentences, entities, page_indices, total_pages,
            layout_regions, config,
        )
        method_label = "combined"

    # Build SummarySentence objects
    scored_sentences = []
    for i, (sent, score) in enumerate(zip(all_sentences, scores)):
        page_num, pos_in_page = page_indices[i]
        scored_sentences.append(SummarySentence(
            text=sent,
            page_num=page_num,
            score=round(score, 6),
            method=method_label,
            bbox=None,
            position_in_page=round(pos_in_page, 4),
        ))

    # Sort by score descending, pick top N
    scored_sentences.sort(key=lambda s: s.score, reverse=True)
    selected = scored_sentences[:config.max_sentences]

    # Re-sort selected sentences by document order (page, then position)
    selected.sort(key=lambda s: (s.page_num, s.position_in_page))

    logger.info(
        "Summarization complete: %d/%d sentences selected (method=%s, doc=%s)",
        len(selected),
        total_sentences,
        method_label,
        document_id or "<unknown>",
    )

    return DocumentSummary(
        sentences=selected,
        document_id=document_id,
        total_pages=total_pages,
        total_sentences=total_sentences,
        entity_summary=_build_entity_summary(entities),
        config_used=_config_to_dict(config),
    )


# ---------------------------------------------------------------------------
# File-based convenience function
# ---------------------------------------------------------------------------


def summarize_from_files(
    text_dir: str,
    entities_path: Optional[str] = None,
    structure_path: Optional[str] = None,
    config: Optional[SummarizationConfig] = None,
) -> DocumentSummary:
    """Summarize a document by loading data from file paths.

    Convenience function that reads page text files from a directory,
    entity data from a JSON file, and layout regions from a structure
    JSON file.

    Args:
        text_dir: Directory containing per-page text files (sorted
            alphabetically).  Each ``.txt`` file is treated as one page.
            Alternatively, if ``text_dir`` points to a single ``.txt``
            file, its content is treated as a single page.
        entities_path: Optional path to a ``.entities.json`` file.
        structure_path: Optional path to a ``structure.json`` file.
        config: Summarization configuration.

    Returns:
        DocumentSummary with selected sentences and metadata.
    """
    # Load page texts
    pages_text = []
    if os.path.isfile(text_dir):
        # Single file mode
        try:
            with open(text_dir, "r", encoding="utf-8") as f:
                pages_text = [f.read()]
        except Exception as exc:
            logger.error("Failed to read text file %s: %s", text_dir, exc)
    elif os.path.isdir(text_dir):
        # Directory mode: each .txt file is a page
        txt_files = sorted(
            f for f in os.listdir(text_dir)
            if f.lower().endswith(".txt")
        )
        for fname in txt_files:
            fpath = os.path.join(text_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    pages_text.append(f.read())
            except Exception as exc:
                logger.warning("Failed to read %s: %s", fpath, exc)
    else:
        logger.error("Text path does not exist: %s", text_dir)

    # Load entities
    entities = []
    if entities_path and os.path.isfile(entities_path):
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both top-level list and {"entities": [...]} format
            if isinstance(data, list):
                entities = data
            elif isinstance(data, dict):
                entities = data.get("entities", [])
        except Exception as exc:
            logger.warning(
                "Failed to load entities from %s: %s", entities_path, exc,
            )

    # Load layout regions
    layout_regions = []
    if structure_path and os.path.isfile(structure_path):
        try:
            with open(structure_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both top-level list and {"pages": [...]} format
            if isinstance(data, dict):
                pages = data.get("pages", [])
                for page in pages:
                    if isinstance(page, dict):
                        regions = page.get("layout_regions", [])
                        layout_regions.extend(regions)
            elif isinstance(data, list):
                layout_regions = data
        except Exception as exc:
            logger.warning(
                "Failed to load structure from %s: %s", structure_path, exc,
            )

    # Derive document_id from directory/file name
    doc_id = os.path.basename(text_dir.rstrip("/\\"))

    return summarize_document(
        pages_text=pages_text,
        entities=entities,
        layout_regions=layout_regions,
        config=config,
        document_id=doc_id,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _config_to_dict(config: SummarizationConfig) -> dict:
    """Convert a SummarizationConfig to a JSON-safe dict."""
    return {
        "method": config.method.value,
        "max_sentences": config.max_sentences,
        "min_sentence_length": config.min_sentence_length,
        "entity_weight": config.entity_weight,
        "position_weight": config.position_weight,
        "textrank_weight": config.textrank_weight,
        "include_metadata": config.include_metadata,
    }


def summary_to_dict(summary: DocumentSummary) -> dict:
    """Convert a DocumentSummary to a JSON-serializable dict.

    Args:
        summary: DocumentSummary instance.

    Returns:
        Dict suitable for ``json.dumps()``.
    """
    return {
        "document_id": summary.document_id,
        "total_pages": summary.total_pages,
        "total_sentences": summary.total_sentences,
        "summary_sentences": [
            {
                "text": s.text,
                "page_num": s.page_num,
                "score": s.score,
                "method": s.method,
                "bbox": s.bbox,
                "position_in_page": s.position_in_page,
            }
            for s in summary.sentences
        ],
        "entity_summary": summary.entity_summary,
        "config_used": summary.config_used,
    }
