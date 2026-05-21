"""Tests for ``ocr_local.translation.quality_estimation`` -- B15.

These tests stub the ``comet`` library entirely so they run on the SDK
CI lane without ``unbabel-comet`` installed.  Coverage:

* :class:`QualityEstimationConfig.from_env` parses every env var with
  type-safe defaults.
* :class:`CometKiwiEstimator` returns ``available=False`` when comet is
  missing -- never raises.
* Air-gapped + missing model raises :class:`ModelNotCachedError`.
* Air-gapped + ``model_path`` pointing at a missing dir raises
  :class:`ModelNotCachedError`.
* :func:`assess_document_quality` aggregates per-span/per-page/document
  scores and threshold counts correctly, even with mixed available/None.
* :func:`enrich_translation_output` is idempotent and adds the right
  shape.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ocr_local.translation.models import (
    DocumentTranslation,
    PageTranslation,
    SpanTranslation,
)
from ocr_local.translation.quality_estimation import (
    CometKiwiEstimator,
    DocumentQualityReport,
    ModelNotCachedError,
    QualityEstimationConfig,
    QualityScore,
    assess_document_quality,
    enrich_translation_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    span_id: str,
    src: str,
    tgt: str,
    page: int = 1,
) -> SpanTranslation:
    return SpanTranslation(
        span_id=span_id,
        source_text=src,
        target_text=tgt,
        source_bbox=[0.0, 0.0, 100.0, 12.0],
        source_bboxes=[[0.0, 0.0, 100.0, 12.0]],
        source_language="en",
        target_language="fr",
        confidence=0.95,
        quality_score=None,
        engine_id="test-engine",
        glossary_hits=[],
    )


def _make_doc(spans_per_page: list[list[SpanTranslation]]) -> DocumentTranslation:
    pages = [
        PageTranslation(page_num=i + 1, spans=spans)
        for i, spans in enumerate(spans_per_page)
    ]
    return DocumentTranslation(
        schema_version="1.1",
        document_id="abc123",
        source_file="x.pdf",
        source_language="en",
        target_language="fr",
        pages=pages,
    )


class _FakeCometModel:
    """Stub mirroring the subset of the comet API we use."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[Any] = []

    def predict(self, payload, batch_size=16, gpus=0):  # noqa: D401
        self.predict_calls.append(
            {"payload": list(payload), "batch_size": batch_size, "gpus": gpus}
        )
        return {"scores": self._scores[: len(payload)]}


def _install_fake_comet(monkeypatch, model: _FakeCometModel | None = None):
    """Replace ``_try_import_comet`` with a stub returning a fake module."""
    fake_module = MagicMock()
    if model is not None:
        fake_module.load_from_checkpoint = MagicMock(return_value=model)
        fake_module.download_model = MagicMock(return_value="/tmp/ckpt")

    def _fake_import():
        if model is None:
            return None
        return fake_module

    monkeypatch.setattr(
        "ocr_local.translation.quality_estimation._try_import_comet",
        _fake_import,
    )
    return fake_module


# ---------------------------------------------------------------------------
# QualityEstimationConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = QualityEstimationConfig()
    assert cfg.enabled is False
    assert cfg.model_id == "Unbabel/wmt22-cometkiwi-da"
    assert cfg.model_path is None
    assert cfg.batch_size == 16
    assert cfg.device == "cpu"
    assert cfg.score_threshold_warn == pytest.approx(0.4)
    assert cfg.score_threshold_reject == pytest.approx(0.2)


def test_config_from_env_parses_all_fields(monkeypatch):
    monkeypatch.setenv("OCR_TRANSLATION_QE_ENABLED", "true")
    monkeypatch.setenv("OCR_TRANSLATION_QE_MODEL_ID", "Custom/model")
    monkeypatch.setenv("OCR_TRANSLATION_QE_MODEL_PATH", "/srv/qe")
    monkeypatch.setenv("OCR_TRANSLATION_QE_BATCH_SIZE", "32")
    monkeypatch.setenv("OCR_TRANSLATION_QE_DEVICE", "cuda")
    monkeypatch.setenv("OCR_TRANSLATION_QE_SCORE_THRESHOLD_WARN", "0.55")
    monkeypatch.setenv("OCR_TRANSLATION_QE_SCORE_THRESHOLD_REJECT", "0.15")
    cfg = QualityEstimationConfig.from_env()
    assert cfg.enabled is True
    assert cfg.model_id == "Custom/model"
    assert cfg.model_path == "/srv/qe"
    assert cfg.batch_size == 32
    assert cfg.device == "cuda"
    assert cfg.score_threshold_warn == pytest.approx(0.55)
    assert cfg.score_threshold_reject == pytest.approx(0.15)


def test_config_from_env_handles_garbage_values(monkeypatch):
    monkeypatch.setenv("OCR_TRANSLATION_QE_BATCH_SIZE", "not-a-number")
    monkeypatch.setenv("OCR_TRANSLATION_QE_SCORE_THRESHOLD_WARN", "")
    cfg = QualityEstimationConfig.from_env()
    assert cfg.batch_size == 16
    assert cfg.score_threshold_warn == pytest.approx(0.4)


def test_config_from_env_empty_model_path_is_none(monkeypatch):
    monkeypatch.setenv("OCR_TRANSLATION_QE_MODEL_PATH", "")
    cfg = QualityEstimationConfig.from_env()
    assert cfg.model_path is None


# ---------------------------------------------------------------------------
# CometKiwiEstimator -- comet absent
# ---------------------------------------------------------------------------


def test_estimator_when_comet_missing_returns_unavailable(monkeypatch):
    _install_fake_comet(monkeypatch, model=None)
    cfg = QualityEstimationConfig(enabled=True)
    estimator = CometKiwiEstimator(cfg)
    result = estimator.score_pair("hello", "bonjour")
    assert isinstance(result, QualityScore)
    assert result.available is False
    assert result.score is None
    assert result.reason == "comet_not_installed"
    assert result.model_id == cfg.model_id


def test_estimator_score_batch_when_comet_missing(monkeypatch):
    _install_fake_comet(monkeypatch, model=None)
    estimator = CometKiwiEstimator(QualityEstimationConfig())
    results = estimator.score_batch([("a", "b"), ("c", "d")])
    assert len(results) == 2
    assert all(r.available is False for r in results)
    assert all(r.reason == "comet_not_installed" for r in results)


def test_estimator_score_batch_empty_input_returns_empty(monkeypatch):
    _install_fake_comet(monkeypatch, model=None)
    estimator = CometKiwiEstimator(QualityEstimationConfig())
    assert estimator.score_batch([]) == []


# ---------------------------------------------------------------------------
# Air-gapped guards
# ---------------------------------------------------------------------------


def test_air_gapped_no_model_path_raises(monkeypatch):
    """allow_download=False AND no model_path -> ModelNotCachedError."""
    fake_model = _FakeCometModel(scores=[0.9])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=None)
    with pytest.raises(ModelNotCachedError) as excinfo:
        CometKiwiEstimator(cfg, allow_download=False)
    assert excinfo.value.engine == "cometkiwi"


def test_air_gapped_missing_model_path_raises(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[0.9])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path / "does-not-exist"))
    with pytest.raises(ModelNotCachedError):
        CometKiwiEstimator(cfg, allow_download=False)


def test_air_gapped_existing_model_path_loads(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[0.85])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    score = estimator.score_pair("hello", "bonjour")
    assert score.available is True
    assert score.score == pytest.approx(0.85)


def test_air_gapped_standalone_model_dir_loads(monkeypatch, tmp_path):
    """Standalone COMETKiwi bundles use load.py instead of a .ckpt path."""
    _install_fake_comet(monkeypatch, model=_FakeCometModel(scores=[0.1]))
    (tmp_path / "state_dict.pt").write_bytes(b"placeholder")
    (tmp_path / "load.py").write_text(
        "\n".join(
            [
                "class _Model:",
                "    def predict(self, payload, batch_size=16, gpus=0):",
                "        return {'scores': [0.77 for _ in payload]}",
                "def load_model(folder):",
                "    return _Model()",
            ]
        ),
        encoding="utf-8",
    )
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    score = estimator.score_pair("hello", "bonjour")
    assert score.available is True
    assert score.score == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# CometKiwiEstimator -- happy path with fake model
# ---------------------------------------------------------------------------


def test_score_pair_returns_clamped_score(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[1.5])  # over-range -> clamp to 1.0
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    score = estimator.score_pair("hello", "bonjour")
    assert score.available is True
    assert score.score == pytest.approx(1.0)


def test_score_pair_negative_clamped_to_zero(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[-0.3])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    assert estimator.score_pair("a", "b").score == pytest.approx(0.0)


def test_score_pair_empty_input_short_circuits(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[0.5])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    score = estimator.score_pair("", "bonjour")
    assert score.available is False
    assert score.reason == "empty_input"
    # The model should NOT have been touched.
    assert fake_model.predict_calls == []


def test_score_batch_handles_mixed_inputs(monkeypatch, tmp_path):
    """Empty inputs receive available=False without disturbing other scores."""
    fake_model = _FakeCometModel(scores=[0.7, 0.3])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    pairs = [("a", "b"), ("", "c"), ("d", "e")]
    results = estimator.score_batch(pairs)
    assert len(results) == 3
    assert results[0].available is True and results[0].score == pytest.approx(0.7)
    assert results[1].available is False
    assert results[1].reason == "empty_input"
    assert results[2].available is True and results[2].score == pytest.approx(0.3)
    # Only the two non-empty pairs were forwarded to the model.
    assert len(fake_model.predict_calls) == 1
    assert len(fake_model.predict_calls[0]["payload"]) == 2


def test_score_batch_uses_configured_batch_size(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[0.5, 0.6])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path), batch_size=4)
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    estimator.score_batch([("a", "b"), ("c", "d")])
    assert fake_model.predict_calls[0]["batch_size"] == 4


def test_score_pair_predict_result_with_scores_attr(monkeypatch, tmp_path):
    """Real comet returns a Prediction with .scores; we tolerate that shape."""
    class _Pred:
        scores = [0.42]

    class _Model:
        def predict(self, payload, batch_size=16, gpus=0):
            return _Pred()

    fake_module = MagicMock()
    fake_module.load_from_checkpoint = MagicMock(return_value=_Model())
    monkeypatch.setattr(
        "ocr_local.translation.quality_estimation._try_import_comet",
        lambda: fake_module,
    )
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    assert estimator.score_pair("a", "b").score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# assess_document_quality
# ---------------------------------------------------------------------------


def test_assess_document_quality_aggregates(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[0.9, 0.5, 0.1])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(
        model_path=str(tmp_path),
        score_threshold_warn=0.6,
        score_threshold_reject=0.3,
    )
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    doc = _make_doc(
        [
            [_make_span("s1", "hello", "bonjour"), _make_span("s2", "world", "monde")],
            [_make_span("s3", "good", "bon")],
        ]
    )

    report = assess_document_quality(doc, estimator)

    assert isinstance(report, DocumentQualityReport)
    assert report.span_count == 3
    assert report.scored_count == 3
    assert report.score_mean == pytest.approx((0.9 + 0.5 + 0.1) / 3)
    assert report.score_min == pytest.approx(0.1)
    # 0.5 < 0.6 (warn) but >= 0.3 (reject) -> warn bucket
    assert report.threshold_warn_count == 1
    # 0.1 < 0.3 (reject)
    assert report.threshold_reject_count == 1
    # Page 1 mean = (0.9 + 0.5) / 2; Page 2 mean = 0.1
    assert report.page_means[1] == pytest.approx(0.7)
    assert report.page_means[2] == pytest.approx(0.1)
    assert len(report.per_span_scores) == 3
    assert {entry["page_num"] for entry in report.per_span_scores} == {1, 2}


def test_assess_document_quality_handles_unavailable_spans(monkeypatch, tmp_path):
    """Empty-input spans appear in per_span_scores but don't pollute aggregates."""
    fake_model = _FakeCometModel(scores=[0.8])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    doc = _make_doc(
        [
            [
                _make_span("s1", "hi", "salut"),
                _make_span("s2", "", "monde"),  # empty source -> unavailable
            ]
        ]
    )

    report = assess_document_quality(doc, estimator)
    assert report.span_count == 2
    assert report.scored_count == 1
    assert report.score_mean == pytest.approx(0.8)
    assert report.score_min == pytest.approx(0.8)
    # Unavailable span recorded with score=None.
    unavailable = [s for s in report.per_span_scores if not s["available"]]
    assert len(unavailable) == 1
    assert unavailable[0]["score"] is None


def test_assess_document_quality_empty_doc(monkeypatch, tmp_path):
    fake_model = _FakeCometModel(scores=[])
    _install_fake_comet(monkeypatch, model=fake_model)
    cfg = QualityEstimationConfig(model_path=str(tmp_path))
    estimator = CometKiwiEstimator(cfg, allow_download=False)
    doc = _make_doc([[]])
    report = assess_document_quality(doc, estimator)
    assert report.span_count == 0
    assert report.scored_count == 0
    assert report.score_mean is None
    assert report.score_min is None
    assert report.per_span_scores == []


def test_assess_document_quality_when_comet_missing(monkeypatch):
    _install_fake_comet(monkeypatch, model=None)
    estimator = CometKiwiEstimator(QualityEstimationConfig())
    doc = _make_doc([[_make_span("s1", "hi", "salut")]])
    report = assess_document_quality(doc, estimator)
    assert report.span_count == 1
    assert report.scored_count == 0
    assert report.score_mean is None
    assert report.per_span_scores[0]["available"] is False


# ---------------------------------------------------------------------------
# enrich_translation_output
# ---------------------------------------------------------------------------


def test_enrich_translation_output_adds_quality_block():
    payload = {"schema_version": "1.1", "document_id": "x"}
    report = DocumentQualityReport(
        model_id="Unbabel/wmt22-cometkiwi-da",
        score_mean=0.7,
        score_min=0.4,
        threshold_warn=0.5,
        threshold_reject=0.2,
        threshold_warn_count=1,
        threshold_reject_count=0,
        per_span_scores=[
            {
                "page_num": 1,
                "span_index": 0,
                "span_id": "s1",
                "score": 0.7,
                "available": True,
                "reason": None,
            }
        ],
        page_means={1: 0.7},
        span_count=1,
        scored_count=1,
    )
    out = enrich_translation_output(payload, report)
    assert "quality_estimation" in out
    qe = out["quality_estimation"]
    assert qe["model_id"] == "Unbabel/wmt22-cometkiwi-da"
    assert qe["score_mean"] == pytest.approx(0.7)
    assert qe["threshold_warn"] == pytest.approx(0.5)
    assert qe["threshold_warn_count"] == 1
    assert qe["page_means"] == {"1": 0.7}
    assert len(qe["per_span_scores"]) == 1


def test_enrich_translation_output_idempotent():
    payload = {"schema_version": "1.1"}
    report = DocumentQualityReport(
        model_id="m",
        score_mean=None,
        score_min=None,
        threshold_warn=0.4,
        threshold_reject=0.2,
        threshold_warn_count=0,
        threshold_reject_count=0,
        per_span_scores=[],
        page_means={},
        span_count=0,
        scored_count=0,
    )
    enrich_translation_output(payload, report)
    first = dict(payload["quality_estimation"])
    enrich_translation_output(payload, report)
    second = dict(payload["quality_estimation"])
    assert first == second


def test_enrich_translation_output_does_not_touch_existing_quality():
    payload = {
        "quality": {"mean_score": 0.5, "below_threshold_count": 0, "quality_class": "draft"},
    }
    report = DocumentQualityReport(
        model_id="m",
        score_mean=0.9,
        score_min=0.9,
        threshold_warn=0.4,
        threshold_reject=0.2,
        threshold_warn_count=0,
        threshold_reject_count=0,
        per_span_scores=[],
        page_means={},
        span_count=0,
        scored_count=0,
    )
    enrich_translation_output(payload, report)
    # ``quality`` (engine self-report) is preserved unchanged.
    assert payload["quality"]["mean_score"] == pytest.approx(0.5)
    assert payload["quality_estimation"]["score_mean"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# quality_custody helper
# ---------------------------------------------------------------------------


def test_emit_quality_estimated_calls_chain():
    from ocr_local.translation.quality_custody import emit_quality_estimated

    chain = MagicMock()
    report = DocumentQualityReport(
        model_id="m",
        score_mean=0.6,
        score_min=0.4,
        threshold_warn=0.5,
        threshold_reject=0.2,
        threshold_warn_count=1,
        threshold_reject_count=0,
        per_span_scores=[],
        page_means={},
        span_count=2,
        scored_count=2,
    )
    emit_quality_estimated(
        chain,
        report,
        document_id="doc1",
        tenant_id="tenant-a",
        target_language="fr",
    )
    assert chain.log_event.called
    args = chain.log_event.call_args
    payload = args.args[1]
    assert payload["document_id"] == "doc1"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["model_id"] == "m"
    assert payload["score_mean"] == pytest.approx(0.6)
    assert payload["threshold_warn_count"] == 1
    # Until B17 lands, fallback path stuffs subtype into payload.
    assert (
        payload.get("event_subtype") == "QUALITY_ESTIMATED"
        or args.args[0] == "TRANSLATION_QUALITY_ESTIMATED"
    )


def test_emit_quality_estimated_no_chain_is_noop():
    from ocr_local.translation.quality_custody import emit_quality_estimated

    report = DocumentQualityReport(
        model_id="m",
        score_mean=None,
        score_min=None,
        threshold_warn=0.4,
        threshold_reject=0.2,
        threshold_warn_count=0,
        threshold_reject_count=0,
        per_span_scores=[],
        page_means={},
        span_count=0,
        scored_count=0,
    )
    # Must not raise.
    emit_quality_estimated(
        None, report, document_id="d", tenant_id="t", target_language="fr"
    )


def test_emit_quality_estimated_swallows_chain_errors():
    from ocr_local.translation.quality_custody import emit_quality_estimated

    chain = MagicMock()
    chain.log_event.side_effect = RuntimeError("boom")
    report = DocumentQualityReport(
        model_id="m",
        score_mean=None,
        score_min=None,
        threshold_warn=0.4,
        threshold_reject=0.2,
        threshold_warn_count=0,
        threshold_reject_count=0,
        per_span_scores=[],
        page_means={},
        span_count=0,
        scored_count=0,
    )
    # Must not raise.
    emit_quality_estimated(
        chain, report, document_id="d", tenant_id="t", target_language="fr"
    )
