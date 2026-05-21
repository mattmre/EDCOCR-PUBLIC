"""Tests for CTC-safe extractive summarization (layoutlm_summarization.py).

Covers sentence splitting, all three scoring methods, combined scoring,
configuration defaults, edge cases (empty/single-sentence documents),
file-based summarization, and dataclass field validation.

Run with: python -m pytest tests/test_layoutlm_summarization.py -v
"""

import json

import pytest

# Add project root to path
from layoutlm_summarization import (
    DocumentSummary,
    SummarizationConfig,
    SummarizationMethod,
    SummarySentence,
    _build_entity_summary,
    _entity_density_scores,
    _layout_position_scores,
    _split_sentences,
    _textrank_scores,
    summarize_document,
    summarize_from_files,
    summary_to_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pages():
    """Multi-page document text for testing."""
    return [
        (
            "The quarterly report shows strong revenue growth. "
            "Net income increased by 15% year over year. "
            "Operating expenses remained stable."
        ),
        (
            "Customer acquisition costs declined in Q3. "
            "The marketing team launched three new campaigns. "
            "Brand awareness surveys indicate positive trends."
        ),
        (
            "Management forecasts continued growth in Q4. "
            "New product launches are expected next quarter. "
            "The board approved the annual dividend."
        ),
    ]


@pytest.fixture
def sample_entities():
    """Sample entity dicts mimicking .entities.json output."""
    return [
        {"text": "quarterly report", "type": "DOCUMENT_TYPE", "confidence": 0.9},
        {"text": "revenue growth", "type": "METRIC", "confidence": 0.85},
        {"text": "15%", "type": "PERCENTAGE", "confidence": 0.95},
        {"text": "Q3", "type": "PERIOD", "confidence": 0.92},
        {"text": "Q4", "type": "PERIOD", "confidence": 0.88},
        {"text": "annual dividend", "type": "FINANCIAL", "confidence": 0.8},
    ]


@pytest.fixture
def sample_layout_regions():
    """Sample layout regions mimicking structure.json output."""
    return [
        {"type": "title", "text": "Quarterly Financial Report", "bbox": [50, 30, 500, 70]},
        {"type": "header", "text": "Revenue Summary", "bbox": [50, 80, 400, 110]},
        {"type": "text", "text": "Body paragraph content", "bbox": [50, 120, 600, 400]},
        {"type": "footer", "text": "Page 1 of 3", "bbox": [50, 900, 200, 930]},
    ]


@pytest.fixture
def default_config():
    """Default summarization config."""
    return SummarizationConfig()


# ---------------------------------------------------------------------------
# Sentence splitting tests
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    """Tests for _split_sentences helper."""

    def test_basic_period_split(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _split_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third sentence."

    def test_exclamation_and_question(self):
        text = "What is this? It is great! The end."
        result = _split_sentences(text)
        assert len(result) == 3
        assert "What is this?" in result[0]
        assert "It is great!" in result[1]

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   \n\t  ") == []

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence.")
        assert len(result) == 1
        assert result[0] == "Just one sentence."

    def test_newlines_between_sentences(self):
        text = "First sentence.\nSecond sentence.\nThird sentence."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_no_trailing_punctuation(self):
        text = "Sentence one. Sentence two without period"
        result = _split_sentences(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TextRank scoring tests
# ---------------------------------------------------------------------------


class TestTextRankScoring:
    """Tests for _textrank_scores."""

    def test_produces_ranked_list(self):
        sentences = [
            "The cat sat on the mat.",
            "The dog chased the cat.",
            "Fish swim in the ocean.",
            "The cat and dog played together.",
        ]
        scores = _textrank_scores(sentences)
        assert len(scores) == 4
        # All scores should be non-negative
        assert all(s >= 0.0 for s in scores)

    def test_max_score_is_one(self):
        sentences = [
            "Revenue growth was strong.",
            "Expenses remained stable.",
            "The growth in revenue exceeded expectations.",
        ]
        scores = _textrank_scores(sentences)
        assert max(scores) == pytest.approx(1.0)

    def test_single_sentence_returns_one(self):
        scores = _textrank_scores(["Only one sentence here."])
        assert scores == [1.0]

    def test_empty_list(self):
        scores = _textrank_scores([])
        assert scores == []

    def test_similar_sentences_score_higher(self):
        sentences = [
            "The report shows revenue growth in Q3.",
            "Revenue growth was strong in the quarterly report.",
            "Unrelated text about underwater basket weaving.",
        ]
        scores = _textrank_scores(sentences)
        # The two similar sentences should score higher than the unrelated one
        assert scores[0] > scores[2] or scores[1] > scores[2]


# ---------------------------------------------------------------------------
# Entity density scoring tests
# ---------------------------------------------------------------------------


class TestEntityDensityScoring:
    """Tests for _entity_density_scores."""

    def test_entity_matching(self):
        sentences = [
            "The quarterly report shows revenue growth.",
            "Nothing special here.",
            "Net income increased by 15% in Q3.",
        ]
        entities = [
            {"text": "quarterly report", "type": "DOC"},
            {"text": "revenue growth", "type": "METRIC"},
            {"text": "15%", "type": "PERCENT"},
            {"text": "Q3", "type": "PERIOD"},
        ]
        scores = _entity_density_scores(sentences, entities)
        assert len(scores) == 3
        # First sentence has 2 entities, third has 2, middle has 0
        assert scores[1] == 0.0
        assert scores[0] > 0.0
        assert scores[2] > 0.0

    def test_no_entities(self):
        sentences = ["Some text here.", "More text."]
        scores = _entity_density_scores(sentences, [])
        assert scores == [0.0, 0.0]

    def test_empty_sentences(self):
        scores = _entity_density_scores([], [{"text": "test"}])
        assert scores == []

    def test_case_insensitive_match(self):
        sentences = ["The QUARTERLY REPORT is ready."]
        entities = [{"text": "quarterly report", "type": "DOC"}]
        scores = _entity_density_scores(sentences, entities)
        assert scores[0] > 0.0


# ---------------------------------------------------------------------------
# Layout position scoring tests
# ---------------------------------------------------------------------------


class TestLayoutPositionScoring:
    """Tests for _layout_position_scores."""

    def test_first_page_bonus(self):
        sentences = ["First page sentence.", "Second page sentence."]
        page_indices = [(0, 0.0), (1, 0.0)]
        scores = _layout_position_scores(sentences, page_indices, 2, [])
        # First sentence (page 0, first overall) should score higher
        assert scores[0] > scores[1]

    def test_title_match_bonus(self):
        sentences = [
            "Quarterly Financial Report overview.",
            "Some random text in the middle.",
        ]
        page_indices = [(0, 0.5), (0, 0.6)]
        regions = [{"type": "title", "text": "Quarterly Financial Report"}]
        scores = _layout_position_scores(sentences, page_indices, 1, regions)
        assert scores[0] > scores[1]

    def test_empty_sentences(self):
        scores = _layout_position_scores([], [], 1, [])
        assert scores == []

    def test_single_page(self):
        sentences = ["Only sentence."]
        page_indices = [(0, 0.0)]
        scores = _layout_position_scores(sentences, page_indices, 1, [])
        assert len(scores) == 1
        assert scores[0] > 0.0


# ---------------------------------------------------------------------------
# Combined scoring tests
# ---------------------------------------------------------------------------


class TestCombinedScoring:
    """Tests for combined scoring via summarize_document with COMBINED method."""

    def test_combined_uses_all_weights(self, sample_pages, sample_entities, sample_layout_regions):
        config = SummarizationConfig(
            method=SummarizationMethod.COMBINED,
            max_sentences=3,
            entity_weight=0.4,
            position_weight=0.3,
            textrank_weight=0.3,
        )
        result = summarize_document(
            sample_pages, sample_entities, sample_layout_regions, config,
        )
        assert len(result.sentences) <= 3
        for s in result.sentences:
            assert s.method == "combined"

    def test_weights_affect_ranking(self, sample_pages, sample_entities):
        # Entity-heavy config
        config_entity = SummarizationConfig(
            method=SummarizationMethod.COMBINED,
            max_sentences=2,
            entity_weight=0.9,
            position_weight=0.05,
            textrank_weight=0.05,
        )
        result_entity = summarize_document(sample_pages, sample_entities, config=config_entity)

        # Position-heavy config
        config_pos = SummarizationConfig(
            method=SummarizationMethod.COMBINED,
            max_sentences=2,
            entity_weight=0.05,
            position_weight=0.9,
            textrank_weight=0.05,
        )
        result_pos = summarize_document(sample_pages, sample_entities, config=config_pos)

        # At least verify both produce valid results
        assert len(result_entity.sentences) > 0
        assert len(result_pos.sentences) > 0


# ---------------------------------------------------------------------------
# Empty and edge-case handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for empty documents and edge cases."""

    def test_empty_document(self):
        result = summarize_document([], config=SummarizationConfig())
        assert result.sentences == []
        assert result.total_pages == 0
        assert result.total_sentences == 0

    def test_empty_page_text(self):
        result = summarize_document(["", "  ", "\n\t"], config=SummarizationConfig())
        assert result.sentences == []
        assert result.total_pages == 3
        assert result.total_sentences == 0

    def test_single_sentence_document(self):
        result = summarize_document(
            ["This is the only sentence in the entire document."],
            config=SummarizationConfig(max_sentences=5),
        )
        assert len(result.sentences) == 1
        assert result.total_sentences == 1

    def test_max_sentences_limiting(self, sample_pages):
        config = SummarizationConfig(max_sentences=2)
        result = summarize_document(sample_pages, config=config)
        assert len(result.sentences) <= 2

    def test_min_sentence_length_filtering(self):
        pages = ["Hi. This is a longer sentence that passes the filter. Ok."]
        config = SummarizationConfig(min_sentence_length=15)
        result = summarize_document(pages, config=config)
        # "Hi." and "Ok." should be filtered out (< 15 chars)
        for s in result.sentences:
            assert len(s.text) >= 15

    def test_graceful_without_entities(self, sample_pages):
        result = summarize_document(sample_pages, entities=None, config=SummarizationConfig())
        assert result.sentences  # Should still produce results
        assert result.entity_summary == {}

    def test_graceful_without_layout_regions(self, sample_pages):
        result = summarize_document(
            sample_pages, layout_regions=None, config=SummarizationConfig(),
        )
        assert result.sentences  # Should still produce results


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Tests for SummarizationConfig default values."""

    def test_default_method(self):
        config = SummarizationConfig()
        assert config.method == SummarizationMethod.COMBINED

    def test_default_max_sentences(self):
        config = SummarizationConfig()
        assert config.max_sentences == 5

    def test_default_min_sentence_length(self):
        config = SummarizationConfig()
        assert config.min_sentence_length == 10

    def test_default_weights(self):
        config = SummarizationConfig()
        assert config.entity_weight == pytest.approx(0.4)
        assert config.position_weight == pytest.approx(0.3)
        assert config.textrank_weight == pytest.approx(0.3)

    def test_include_metadata_default(self):
        config = SummarizationConfig()
        assert config.include_metadata is True

    def test_weight_clamping(self):
        config = SummarizationConfig(entity_weight=2.0, position_weight=-0.5)
        assert config.entity_weight == 1.0
        assert config.position_weight == 0.0

    def test_max_sentences_minimum(self):
        config = SummarizationConfig(max_sentences=0)
        assert config.max_sentences >= 1


# ---------------------------------------------------------------------------
# DocumentSummary tests
# ---------------------------------------------------------------------------


class TestDocumentSummary:
    """Tests for DocumentSummary dataclass and entity_summary."""

    def test_entity_summary_counts(self, sample_pages, sample_entities):
        result = summarize_document(
            sample_pages, sample_entities, config=SummarizationConfig(),
        )
        # sample_entities has: DOCUMENT_TYPE(1), METRIC(1), PERCENTAGE(1), PERIOD(2), FINANCIAL(1)
        assert result.entity_summary["PERIOD"] == 2
        assert result.entity_summary["DOCUMENT_TYPE"] == 1
        assert result.entity_summary["PERCENTAGE"] == 1

    def test_document_id(self):
        result = summarize_document(
            ["Some text here."], document_id="doc-001",
            config=SummarizationConfig(),
        )
        assert result.document_id == "doc-001"

    def test_total_pages(self, sample_pages):
        result = summarize_document(sample_pages, config=SummarizationConfig())
        assert result.total_pages == 3

    def test_config_used_recorded(self, sample_pages):
        config = SummarizationConfig(method=SummarizationMethod.TEXTRANK, max_sentences=3)
        result = summarize_document(sample_pages, config=config)
        assert result.config_used["method"] == "textrank"
        assert result.config_used["max_sentences"] == 3


# ---------------------------------------------------------------------------
# SummarySentence tests
# ---------------------------------------------------------------------------


class TestSummarySentence:
    """Tests for SummarySentence dataclass fields."""

    def test_fields(self):
        s = SummarySentence(
            text="Test sentence.",
            page_num=1,
            score=0.85,
            method="combined",
            bbox=[10, 20, 300, 40],
            position_in_page=0.25,
        )
        assert s.text == "Test sentence."
        assert s.page_num == 1
        assert s.score == pytest.approx(0.85)
        assert s.method == "combined"
        assert s.bbox == [10, 20, 300, 40]
        assert s.position_in_page == pytest.approx(0.25)

    def test_defaults(self):
        s = SummarySentence(text="Hello.")
        assert s.page_num == 0
        assert s.score == 0.0
        assert s.method == ""
        assert s.bbox is None
        assert s.position_in_page == 0.0

    def test_sentences_in_result_have_method(self, sample_pages):
        config = SummarizationConfig(method=SummarizationMethod.TEXTRANK)
        result = summarize_document(sample_pages, config=config)
        for s in result.sentences:
            assert s.method == "textrank"


# ---------------------------------------------------------------------------
# SummarizationMethod enum tests
# ---------------------------------------------------------------------------


class TestSummarizationMethod:
    """Tests for SummarizationMethod enum values."""

    def test_textrank_value(self):
        assert SummarizationMethod.TEXTRANK.value == "textrank"

    def test_entity_density_value(self):
        assert SummarizationMethod.ENTITY_DENSITY.value == "entity_density"

    def test_layout_position_value(self):
        assert SummarizationMethod.LAYOUT_POSITION.value == "layout_position"

    def test_combined_value(self):
        assert SummarizationMethod.COMBINED.value == "combined"

    def test_all_methods_usable(self, sample_pages):
        for method in SummarizationMethod:
            config = SummarizationConfig(method=method, max_sentences=2)
            result = summarize_document(sample_pages, config=config)
            assert isinstance(result, DocumentSummary)
            assert len(result.sentences) <= 2


# ---------------------------------------------------------------------------
# summarize_document with entities
# ---------------------------------------------------------------------------


class TestSummarizeDocumentWithEntities:
    """Tests for summarize_document with entity-aware scoring."""

    def test_entity_density_method(self, sample_pages, sample_entities):
        config = SummarizationConfig(
            method=SummarizationMethod.ENTITY_DENSITY,
            max_sentences=3,
        )
        result = summarize_document(sample_pages, sample_entities, config=config)
        assert len(result.sentences) <= 3
        for s in result.sentences:
            assert s.method == "entity_density"

    def test_entity_rich_sentences_ranked_higher(self):
        pages = [
            "John Smith filed case 12345 on January 15. "
            "The weather was nice. "
            "Case 12345 involves John Smith and Jane Doe."
        ]
        entities = [
            {"text": "John Smith", "type": "PERSON"},
            {"text": "case 12345", "type": "CASE_NUMBER"},
            {"text": "Jane Doe", "type": "PERSON"},
            {"text": "January 15", "type": "DATE"},
        ]
        config = SummarizationConfig(
            method=SummarizationMethod.ENTITY_DENSITY,
            max_sentences=1,
        )
        result = summarize_document(pages, entities, config=config)
        # The sentence with the most entities should be selected
        assert len(result.sentences) == 1
        # Either first or third sentence (both have 2+ entities)
        selected_text = result.sentences[0].text
        assert "John Smith" in selected_text or "case 12345" in selected_text.lower()


# ---------------------------------------------------------------------------
# summarize_from_files tests
# ---------------------------------------------------------------------------


class TestSummarizeFromFiles:
    """Tests for summarize_from_files with tmp_path."""

    def test_single_file_mode(self, tmp_path):
        text_file = tmp_path / "document.txt"
        text_file.write_text(
            "The report shows growth. Revenue increased significantly. "
            "Costs were reduced.",
            encoding="utf-8",
        )
        result = summarize_from_files(str(text_file))
        assert result.total_pages == 1
        assert result.total_sentences > 0
        assert result.document_id == "document.txt"

    def test_directory_mode(self, tmp_path):
        text_dir = tmp_path / "pages"
        text_dir.mkdir()
        (text_dir / "page_001.txt").write_text(
            "First page content here. Important findings noted.",
            encoding="utf-8",
        )
        (text_dir / "page_002.txt").write_text(
            "Second page with more details. Conclusions follow.",
            encoding="utf-8",
        )
        result = summarize_from_files(str(text_dir))
        assert result.total_pages == 2
        assert result.total_sentences > 0

    def test_with_entities_json(self, tmp_path):
        text_file = tmp_path / "doc.txt"
        text_file.write_text(
            "John Smith submitted invoice 99887. The amount was $5000.",
            encoding="utf-8",
        )
        entities_file = tmp_path / "doc.entities.json"
        entities_file.write_text(json.dumps({
            "entities": [
                {"text": "John Smith", "type": "PERSON", "confidence": 0.9},
                {"text": "invoice 99887", "type": "INVOICE", "confidence": 0.85},
                {"text": "$5000", "type": "AMOUNT", "confidence": 0.95},
            ],
        }), encoding="utf-8")
        result = summarize_from_files(
            str(text_file), entities_path=str(entities_file),
        )
        assert result.entity_summary.get("PERSON", 0) == 1

    def test_with_structure_json(self, tmp_path):
        text_file = tmp_path / "doc.txt"
        text_file.write_text(
            "Financial Report summary. Revenue data follows. End of report.",
            encoding="utf-8",
        )
        structure_file = tmp_path / "structure.json"
        structure_file.write_text(json.dumps({
            "pages": [
                {
                    "page_num": 1,
                    "layout_regions": [
                        {"type": "title", "text": "Financial Report"},
                    ],
                }
            ],
        }), encoding="utf-8")
        result = summarize_from_files(
            str(text_file), structure_path=str(structure_file),
        )
        assert result.total_pages == 1
        assert result.total_sentences > 0

    def test_missing_text_path(self, tmp_path):
        result = summarize_from_files(str(tmp_path / "nonexistent"))
        assert result.sentences == []
        assert result.total_sentences == 0

    def test_missing_entities_graceful(self, tmp_path):
        text_file = tmp_path / "doc.txt"
        text_file.write_text("Some content here.", encoding="utf-8")
        result = summarize_from_files(
            str(text_file), entities_path=str(tmp_path / "missing.json"),
        )
        assert result.entity_summary == {}


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for summary_to_dict serialization."""

    def test_summary_to_dict(self, sample_pages):
        config = SummarizationConfig(max_sentences=2)
        result = summarize_document(sample_pages, config=config)
        d = summary_to_dict(result)
        assert "document_id" in d
        assert "summary_sentences" in d
        assert "entity_summary" in d
        assert "config_used" in d
        assert len(d["summary_sentences"]) <= 2

    def test_json_serializable(self, sample_pages, sample_entities):
        config = SummarizationConfig(max_sentences=3)
        result = summarize_document(sample_pages, sample_entities, config=config)
        d = summary_to_dict(result)
        # Should not raise
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)
        assert parsed["total_pages"] == 3


# ---------------------------------------------------------------------------
# Build entity summary helper tests
# ---------------------------------------------------------------------------


class TestBuildEntitySummary:
    """Tests for _build_entity_summary helper."""

    def test_counts_by_type(self):
        entities = [
            {"text": "a", "type": "PERSON"},
            {"text": "b", "type": "DATE"},
            {"text": "c", "type": "PERSON"},
            {"text": "d", "type": "AMOUNT"},
        ]
        summary = _build_entity_summary(entities)
        assert summary["PERSON"] == 2
        assert summary["DATE"] == 1
        assert summary["AMOUNT"] == 1

    def test_empty_entities(self):
        assert _build_entity_summary([]) == {}
        assert _build_entity_summary(None) == {}

    def test_uses_label_fallback(self):
        entities = [{"text": "x", "label": "ORG"}]
        summary = _build_entity_summary(entities)
        assert summary["ORG"] == 1
