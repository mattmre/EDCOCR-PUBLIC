"""Pure, evidence-preserving language and OCR candidate routing decisions.

``detected_label`` records what detection actually established.  A bootstrap
model used to read an unknown page is not evidence that the page has that
language.  Callers must keep those two fields separate in provenance.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from ocr_local.config.language_config import (
    LANGUAGE_REGISTRY,
    PADDLE_UNSUPPORTED_LANGUAGES,
    get_paddle_model_family,
)

_UNKNOWN = frozenset({"", "und", "unknown"})
_SCRIPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "latin": ("en", "fr"),
    "cyrillic": ("ru", "uk"),
    "arabic": ("ar", "fa"),
    "devanagari": ("hi", "mr"),
    "cjk": ("ch", "japan", "korean"),
    "han": ("ch", "chinese_cht"),
    "kana": ("japan",),
    "japanese": ("japan",),
    "hangul": ("korean",),
    "korean": ("korean",),
    "georgian": ("ka",),
    "tamil": ("ta",),
    "telugu": ("te",),
    "greek": ("el",),
    "bengali": ("bn",),
    "kannada": ("kn",),
    "thai": ("th",),
}


def _unicode_script(text: str) -> str | None:
    """Return the script of a letter without treating OCR language as fact."""
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script in ("latin", "greek", "cyrillic"):
            if name.startswith(script.upper() + " "):
                return script
    return None


def text_supports_script(text: str, script: str) -> bool:
    """Check that recognized letters predominantly belong to the model script."""
    wanted = str(script or "").strip().upper()
    if not wanted:
        return False
    letters = [unicodedata.name(char, "") for char in text if char.isalpha()]
    return len(letters) >= 3 and sum(
        name.startswith(wanted + " ") for name in letters
    ) / len(letters) >= 0.8


def probe_image_script_candidates(
    image,
    *,
    timeout_seconds: float = 3.0,
    max_pixels: int = 2_000_000,
) -> tuple[str, ...]:
    """Corroborate Latin/Greek/Cyrillic script with bounded single-model OCR.

    OSD alone has mistaken Greek for Cyrillic and sparse English for Cyrillic.
    Three independent traineddata probes compete on confidence *and* the
    fraction of letters actually produced in that script. Mixed-language OCR
    is deliberately excluded because its confidence is not a quality rank.
    No result claims a detected language; ambiguous or unavailable probes
    return no script evidence so callers retain a reviewable bootstrap route.
    """
    if not 0 < timeout_seconds <= 10 or not 1 <= max_pixels <= 8_000_000:
        raise ValueError("script probe limits are out of range")
    import pytesseract

    width, height = image.size
    if width <= 0 or height <= 0:
        return ()
    if width * height > max_pixels:
        ratio = math.sqrt(max_pixels / (width * height))
        image = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio)))
        )
    scores: list[tuple[float, str]] = []
    for script, traineddata in (("latin", "eng"), ("greek", "ell"), ("cyrillic", "rus")):
        try:
            data = pytesseract.image_to_data(
                image,
                lang=traineddata,
                config="--psm 11",
                output_type=pytesseract.Output.DICT,
                timeout=timeout_seconds,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        confidence_total = 0.0
        confidence_weight = 0
        script_letters = 0
        all_letters = 0
        for word, raw_confidence in zip(data.get("text", ()), data.get("conf", ())):
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or confidence < 0:
                continue
            for char in str(word):
                if not char.isalpha():
                    continue
                all_letters += 1
                if _unicode_script(char) == script:
                    script_letters += 1
                confidence_total += confidence
                confidence_weight += 1
        if all_letters < 3 or not confidence_weight:
            continue
        support = script_letters / all_letters
        if support < 0.8:
            continue
        scores.append(((confidence_total / confidence_weight) * support, script))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 50:
        return ()
    if len(scores) > 1 and scores[0][0] - scores[1][0] < 5:
        return ()
    return (scores[0][1],)


@dataclass(frozen=True)
class LanguageRoute:
    detected_label: str
    engine: str
    model_language: str  # Canonical registry code for existing checkpoint fields.
    runtime_model: str  # Paddle family or installed Tesseract traineddata code.
    candidate_languages: tuple[str, ...]
    reason: str
    ambiguous: bool
    needs_review: bool


def _registry_code(value: str | None) -> str | None:
    code = str(value or "").strip().lower().replace("_", "-")
    if not code or code in _UNKNOWN:
        return None
    if code in LANGUAGE_REGISTRY:
        return code
    for registry_code, entry in LANGUAGE_REGISTRY.items():
        if code in entry.fasttext_codes or code == entry.bcp47.lower():
            return registry_code
    return None


def _route_known(code: str, detected_label: str, reason: str) -> LanguageRoute:
    if code in PADDLE_UNSUPPORTED_LANGUAGES:
        tess = LANGUAGE_REGISTRY[code].tesseract_code
        return LanguageRoute(
            detected_label, "tesseract", code, tess, (code,), reason, False, False
        )
    # Pass a registry language, not a model family, to existing engine creation:
    # its adapter maps to the correct PaddleOCR 2.x family.
    assert get_paddle_model_family(code) is not None
    return LanguageRoute(
        detected_label,
        "paddle",
        code,
        get_paddle_model_family(code) or code,
        (code,),
        reason,
        False,
        False,
    )


def route_language(
    detected_language: str | None,
    *,
    explicit_hint: str | None = None,
    script_evidence: Iterable[str] = (),
    prior_hint: str | None = None,
) -> LanguageRoute:
    """Choose an admitted OCR model without relabeling uncertain detection.

    Explicit hints must resolve to the local registry.  Script evidence only
    chooses a bounded probe model; it never manufactures a detected language.
    Unknown pages remain eligible for Paddle through a valid ``en`` bootstrap.
    No route here authorizes skipping OCR because embedded text exists.
    """
    raw_detected = str(detected_language or "").strip().lower()
    detected_code = _registry_code(raw_detected)
    detected_label = detected_code or (raw_detected if raw_detected else "und")

    if explicit_hint is not None:
        hinted = _registry_code(explicit_hint)
        if hinted is not None:
            return _route_known(hinted, detected_label, "validated_explicit_hint")
        invalid_hint = True
    else:
        invalid_hint = False

    scripts = tuple(
        dict.fromkeys(
            str(script).strip().lower()
            for script in script_evidence
            if str(script).strip().lower() in _SCRIPT_CANDIDATES
        )
    )
    candidates = tuple(
        dict.fromkeys(
            code
            for script in scripts
            for code in _SCRIPT_CANDIDATES[script]
            if code in LANGUAGE_REGISTRY
        )
    )[:3]
    prior_code = _registry_code(prior_hint)
    if detected_code is None and prior_code is not None and not invalid_hint:
        prior_script = LANGUAGE_REGISTRY[prior_code].script
        if not scripts or prior_script in scripts or not candidates:
            prior_route = _route_known(
                prior_code, detected_label, "unverified_source_prior"
            )
            return LanguageRoute(
                prior_route.detected_label,
                prior_route.engine,
                prior_route.model_language,
                prior_route.runtime_model,
                prior_route.candidate_languages,
                prior_route.reason,
                True,
                True,
            )
    if detected_code is not None:
        route = _route_known(
            detected_code, detected_label, "detected_registry_language"
        )
        detected_script = LANGUAGE_REGISTRY[detected_code].script
        # Script evidence from the current page may contradict a document-level
        # prior.  It chooses a probe model, never a claimed detected language.
        if (
            scripts
            and detected_script not in scripts
            and detected_code not in candidates
            and candidates
        ):
            selected = candidates[0]
            if selected in PADDLE_UNSUPPORTED_LANGUAGES:
                engine = "tesseract"
                runtime_model = LANGUAGE_REGISTRY[selected].tesseract_code
            else:
                engine = "paddle"
                runtime_model = get_paddle_model_family(selected) or selected
            return LanguageRoute(
                detected_label,
                engine,
                selected,
                runtime_model,
                candidates,
                "document_prior_script_conflict",
                True,
                True,
            )
        if not invalid_hint:
            return route
        return LanguageRoute(
            route.detected_label,
            route.engine,
            route.model_language,
            route.runtime_model,
            route.candidate_languages,
            "invalid_hint_ignored;detected_registry_language",
            route.ambiguous,
            True,
        )
    ambiguous = len(scripts) != 1 or len(candidates) > 1
    if not candidates:
        candidates = ("en",)
    selected = candidates[0]
    if selected in PADDLE_UNSUPPORTED_LANGUAGES:
        engine = "tesseract"
        runtime_model = LANGUAGE_REGISTRY[selected].tesseract_code
    else:
        engine = "paddle"
        runtime_model = get_paddle_model_family(selected) or selected
    reason = "script_bootstrap" if scripts else "unknown_language_bootstrap"
    if prior_code is not None and scripts:
        reason = "source_prior_script_conflict;" + reason
    if invalid_hint:
        reason = "invalid_hint_ignored;" + reason
    return LanguageRoute(
        detected_label,
        engine,
        selected,
        runtime_model,
        candidates,
        reason,
        ambiguous,
        True,
    )


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    engine: str
    confidence: float | None = None
    coverage: float | None = None
    script_supported: bool | None = None
    language_support: float | None = None


@dataclass(frozen=True)
class CandidateDecision:
    use_candidate: bool
    reason: str
    needs_review: bool


def _unit_score(value: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 1 else None


def compare_ocr_candidates(
    prior: OCRCandidate, candidate: OCRCandidate
) -> CandidateDecision:
    """Accept a rerun only with comparable, independently corroborated gain.

    Confidence from different engines is not calibrated.  Text length alone is
    not a quality metric.  ``coverage`` is a caller-measured fraction of the
    page's text regions, independent of the engine's confidence estimate.
    """
    if not candidate.text or not candidate.text.strip():
        return CandidateDecision(False, "candidate_empty", bool(prior.text.strip()))
    candidate_confidence = _unit_score(candidate.confidence)
    if candidate_confidence is None or candidate.script_supported is False:
        return CandidateDecision(False, "candidate_invalid_evidence", True)
    if not prior.text or not prior.text.strip():
        return CandidateDecision(
            True, "prior_empty", _unit_score(candidate.coverage) is None
        )
    prior_confidence = _unit_score(prior.confidence)
    prior_coverage = _unit_score(prior.coverage)
    candidate_coverage = _unit_score(candidate.coverage)
    if (
        prior.engine != candidate.engine
        or prior_confidence is None
        or prior_coverage is None
        or candidate_coverage is None
        or candidate.script_supported is not True
    ):
        return CandidateDecision(False, "incomparable_evidence", True)
    if candidate_coverage + 1e-9 < prior_coverage:
        return CandidateDecision(False, "coverage_regressed", True)
    prior_language_support = _unit_score(prior.language_support)
    candidate_language_support = _unit_score(candidate.language_support)
    if (
        prior.engine == candidate.engine == "tesseract"
        and prior_confidence < 0.85 <= candidate_confidence
        and candidate_language_support is not None
        and candidate_language_support >= 0.65
    ):
        return CandidateDecision(True, "supported_tesseract_quality_threshold_crossed", True)
    if (
        prior_language_support is not None
        and candidate_language_support is not None
        and candidate_language_support - prior_language_support >= 0.2
        and candidate_confidence >= prior_confidence - 0.02
    ):
        return CandidateDecision(True, "corroborated_language_support_gain", True)
    # A separately measured recovery of text regions may be decisive even
    # when the detector's mean confidence is merely comparable.  This does
    # not compare confidence across engines or reward raw character count.
    coverage_gain = candidate_coverage - prior_coverage
    if (
        coverage_gain >= max(0.001, 0.2 * prior_coverage)
        and candidate_confidence >= prior_confidence - 0.02
    ):
        return CandidateDecision(True, "corroborated_region_recovery", False)
    if candidate_confidence < prior_confidence + 0.05:
        return CandidateDecision(False, "confidence_gain_insufficient", True)
    return CandidateDecision(True, "corroborated_gain", False)


def compare_cross_engine_candidates(
    prior: OCRCandidate, candidate: OCRCandidate
) -> CandidateDecision:
    """Require independent text-language and visual support across engines.

    Paddle and Tesseract confidence scales cannot be compared. A candidate's
    own confidence only establishes a minimum readable result; independently
    measured coverage and language support decide whether it can replace the
    prior text. Both readings remain review evidence at the call site.
    """
    if prior.engine == candidate.engine:
        return CandidateDecision(False, "same_engine_use_standard_comparison", True)
    if not candidate.text.strip():
        return CandidateDecision(False, "candidate_empty", bool(prior.text.strip()))
    candidate_confidence = _unit_score(candidate.confidence)
    if candidate_confidence is None or candidate_confidence < 0.70 or candidate.script_supported is not True:
        return CandidateDecision(False, "candidate_unqualified", True)
    if not prior.text.strip():
        return CandidateDecision(True, "prior_empty", True)
    prior_coverage = _unit_score(prior.coverage)
    candidate_coverage = _unit_score(candidate.coverage)
    if prior_coverage is None or candidate_coverage is None:
        return CandidateDecision(False, "independent_coverage_unavailable", True)
    if candidate_coverage + 1e-9 < prior_coverage:
        return CandidateDecision(False, "coverage_regressed", True)
    prior_support = _unit_score(prior.language_support)
    candidate_support = _unit_score(candidate.language_support)
    prior_confidence = _unit_score(prior.confidence)
    if (
        prior_confidence is not None
        and prior_confidence < 0.55
        and candidate_confidence >= 0.90
        and candidate_coverage >= 0.90
        and candidate_support is not None
        and candidate_support >= 0.60
        and (prior_support is None or candidate_support - prior_support >= 0.20)
    ):
        return CandidateDecision(True, "weak_primary_independently_corroborated", True)
    if candidate_support is None or candidate_support < 0.65:
        return CandidateDecision(False, "candidate_language_unsupported", True)
    if prior_support is not None and candidate_support - prior_support >= 0.20:
        return CandidateDecision(True, "cross_engine_language_support_gain", True)
    if candidate_coverage - prior_coverage >= max(0.03, 0.15 * prior_coverage):
        return CandidateDecision(True, "cross_engine_region_recovery", True)
    return CandidateDecision(False, "no_independent_cross_engine_gain", True)


def _word_rect(box) -> tuple[float, float, float, float] | None:
    """Read a PDF-point word rectangle or quad without repairing bad geometry."""
    try:
        if len(box) != 4:
            return None
        if hasattr(box[0], "__iter__"):
            points = [(float(point[0]), float(point[1])) for point in box]
            xs, ys = zip(*points)
            rect = (min(xs), min(ys), max(xs), max(ys))
        else:
            rect = tuple(float(value) for value in box)
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in rect):
        return None
    x0, y0, x1, y1 = rect
    return rect if x1 > x0 and y1 > y0 else None


def _rect_iou(left, right) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def compare_grounded_word_candidates(
    prior: OCRCandidate,
    candidate: OCRCandidate,
    *,
    prior_lines,
    candidate_lines,
) -> CandidateDecision:
    """Select a same-engine word correction grounded to the same page region.

    A page mean may fall because unchanged words receive different scores at a
    different DPI. Only a changed, previously uncertain word can justify the
    replacement; every word must align in physical PDF coordinates. This does
    not infer spelling from a language model or merge two readings.
    """
    if (
        str(prior.engine).casefold() != "tesseract"
        or str(candidate.engine).casefold() != "tesseract"
    ):
        return CandidateDecision(False, "not_same_tesseract_engine", True)
    old_coverage = _unit_score(prior.coverage)
    new_coverage = _unit_score(candidate.coverage)
    support = _unit_score(candidate.language_support)
    if (
        old_coverage is None or new_coverage is None
        or new_coverage + 1e-9 < old_coverage
        or candidate.script_supported is not True
        or support is None or support < 0.65
    ):
        return CandidateDecision(False, "independent_support_unavailable", True)
    if not prior_lines or len(prior_lines) != len(candidate_lines):
        return CandidateDecision(False, "word_alignment_incomplete", True)
    improved = 0
    for old, new in zip(prior_lines, candidate_lines):
        try:
            old_text, old_box, old_score = old
            new_text, new_box, new_score = new
        except (TypeError, ValueError):
            return CandidateDecision(False, "word_evidence_malformed", True)
        old_rect, new_rect = _word_rect(old_box), _word_rect(new_box)
        old_conf, new_conf = _unit_score(old_score), _unit_score(new_score)
        if (
            old_rect is None or new_rect is None
            or old_conf is None or new_conf is None
            or _rect_iou(old_rect, new_rect) < 0.5
        ):
            return CandidateDecision(False, "word_alignment_unverified", True)
        if str(old_text) == str(new_text):
            continue
        if not str(old_text).strip() or not str(new_text).strip():
            return CandidateDecision(False, "word_added_or_removed", True)
        if old_conf >= 0.70 or new_conf < 0.70 or new_conf - old_conf < 0.15:
            return CandidateDecision(False, "changed_word_not_corroborated", True)
        improved += 1
    if not improved:
        return CandidateDecision(False, "no_changed_word_gain", True)
    return CandidateDecision(True, "grounded_uncertain_word_recovered", True)
