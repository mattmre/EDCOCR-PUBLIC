#!/usr/bin/env python3
"""Disprove OCR rescue outputs against independent visible-text and geometry truth.

Input mapping JSON: ``{"case_id": {"pdf": "...", "text": "...", "metadata": {...}}}``.
All paths in the mapping are resolved relative to that mapping file.  A failed
assertion makes the process exit nonzero even though a JSON report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image

RENDER_DPI = 120
_RUNTIME_MODELS_BY_LANGUAGE = {
    "en": frozenset({"en", "eng"}),
    "fr": frozenset({"latin", "fra"}),
    "el": frozenset({"ell"}),
}
_API_RUNTIME_MODELS = {
    "paddle": {"en": "en", "fr": "latin"},
    "tesseract": {"en": "eng", "fr": "fra", "el": "ell"},
}
_UNCERTAIN_LANGUAGE_ROUTES = frozenset({
    "unknown_language_bootstrap", "script_bootstrap",
    "per_page_model_candidate", "validated_explicit_hint",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    """Preserve every script and punctuation mark; normalize only presentation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, start=1):
        current = [row]
        for column, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def _ratio(expected: str, observed: str, *, words: bool = False) -> float:
    if words:
        expected_units = re.findall(r"\w+|[^\w\s]", expected, re.UNICODE)
        observed_units = re.findall(r"\w+|[^\w\s]", observed, re.UNICODE)
    else:
        expected_units = expected
        observed_units = observed
    if not expected_units:
        return 0.0 if not observed_units else 1.0
    return edit_distance(expected_units, observed_units) / len(expected_units)


def _page_texts(path: Path, page_count: int, metadata: dict | None = None) -> list[str]:
    value = path.read_text(encoding="utf-8")
    lengths = (metadata or {}).get("text_page_lengths")
    if lengths is not None:
        if not isinstance(lengths, list) or len(lengths) != page_count or any(isinstance(length, bool) or not isinstance(length, int) or length < 0 for length in lengths):
            raise ValueError("invalid observed page text lengths")
        pieces = []
        cursor = 0
        for index, length in enumerate(lengths):
            pieces.append(value[cursor : cursor + length])
            cursor += length
            if index + 1 < page_count:
                if value[cursor : cursor + 2] != "\n\n":
                    raise ValueError("coordinator TXT page separator differs from verified span framing")
                cursor += 2
        if cursor != len(value):
            raise ValueError("coordinator TXT has unframed trailing content")
        return pieces
    if page_count == 1 and "\f" not in value:
        return [value]
    pieces = value.split("\f")
    if pieces and not pieces[-1].strip():
        pieces.pop()
    if len(pieces) != page_count:
        raise ValueError("TXT sidecar page count/framing differs from PDF")
    return pieces


def _render(page: fitz.Page) -> np.ndarray:
    scale = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()[:, :, :3]


def _render_image(path: Path, size_pt: list[float]) -> np.ndarray:
    with Image.open(path) as original:
        image = original.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        white.alpha_composite(image)
        # Reconstruct the independent source view in PDF point space, then
        # rasterize with the same renderer used for output PDFs. PNG stores
        # DPI as integer pixels/metre, so its physical size can differ by a
        # fraction of a point from rounded corpus truth. Direct rounding of
        # truth dimensions yields a 1020x1320 array while fitz correctly
        # renders that source's 612.001x792.002 point page at 1021x1321.
        source_dpi = original.info.get("dpi") or (300.0, 300.0)
        if not isinstance(source_dpi, (tuple, list)) or len(source_dpi) != 2:
            raise ValueError("source image DPI must contain X and Y values")
        dpi_x, dpi_y = (float(value) for value in source_dpi)
        if not all(math.isfinite(value) and 10 <= value <= 2400 for value in (dpi_x, dpi_y)):
            raise ValueError("source image DPI is outside the physical-size bound")
        width_pt = image.width * 72.0 / dpi_x
        height_pt = image.height * 72.0 / dpi_y
        if max(
            abs(width_pt - size_pt[0]),
            abs(height_pt - size_pt[1]),
        ) > 0.1:
            raise ValueError("source image DPI disagrees with corpus physical size")
        encoded = io.BytesIO()
        white.convert("RGB").save(encoded, format="PNG")
        with fitz.open() as source_view:
            page = source_view.new_page(
                width=width_pt, height=height_pt
            )
            page.insert_image(page.rect, stream=encoded.getvalue())
            return _render(page)


def visual_metrics(source: np.ndarray, output_page: fitz.Page) -> dict[str, float]:
    output = _render(output_page)
    if source.shape != output.shape:
        raise ValueError("render dimensions differ at equal physical scale")
    mae = float(np.mean(np.abs(source.astype(np.int16) - output.astype(np.int16))) / 255.0)
    source_fg = np.mean(source, axis=2) < 215
    output_fg = np.mean(output, axis=2) < 215
    if source_fg.any():
        padded = np.pad(output_fg, 1)
        nearby = np.zeros_like(output_fg)
        for y in range(3):
            for x in range(3):
                nearby |= padded[y : y + output_fg.shape[0], x : x + output_fg.shape[1]]
        recall = float(np.count_nonzero(source_fg & nearby) / np.count_nonzero(source_fg))
    else:
        recall = 1.0 if not output_fg.any() else 0.0
    return {"rgb_mae": round(mae, 6), "foreground_recall_1px": round(recall, 6)}


def _overlap_fraction(expected: fitz.Rect, actual: fitz.Rect) -> float:
    intersection = expected & actual
    if expected.is_empty or actual.is_empty or intersection.is_empty:
        return 0.0
    return max(0.0, intersection.get_area() / expected.get_area())


def _metadata(value: Any, base: Path) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        path = (base / value).resolve()
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _page_metadata(metadata: dict, index: int, page_count: int) -> dict:
    pages = metadata.get("pages")
    if isinstance(pages, list) and len(pages) == page_count and isinstance(pages[index], dict):
        return pages[index]
    if page_count == 1 and pages is None:
        return metadata
    return {}


def _language_provenance_disposition(expected: str | None, page_metadata: dict) -> str:
    """Keep an unknown detector label truthful while auditing model choice."""
    if not expected:
        return "not_applicable"
    detected = page_metadata.get("detected_language")
    if detected == expected:
        return "confirmed_detection"
    if detected != "und":
        return "invalid"
    if (
        page_metadata.get("needs_review") is True
        and page_metadata.get("assessment_state") == "assessed"
        and page_metadata.get("status") in {"ok", "fallback"}
        and page_metadata.get("route_reason") in _UNCERTAIN_LANGUAGE_ROUTES
        and page_metadata.get("model_language") == expected
        and page_metadata.get("runtime_model") in _RUNTIME_MODELS_BY_LANGUAGE.get(expected, ())
        and isinstance(page_metadata.get("candidate_disposition"), str)
        and bool(page_metadata["candidate_disposition"])
    ):
        return "reviewable_unknown_detection"
    # Standalone API provenance records the route and a fresh but unconfirmed
    # text-language candidate as review reasons, rather than coordinator's
    # assessment/candidate fields. Require both exact structured phrases and
    # a measured source coverage assessment; a generic review flag is not
    # evidence of language routing.
    reasons = page_metadata.get("review_reasons")
    coverage = page_metadata.get("independent_coverage")
    if (
        page_metadata.get("needs_review") is True
        and page_metadata.get("status") in {"ok", "fallback"}
        and page_metadata.get("source_prior_label") in {"", "und"}
        and page_metadata.get("model_language") == expected
        and page_metadata.get("runtime_model") == _API_RUNTIME_MODELS.get(
            page_metadata.get("inference_backend"), {}
        ).get(expected)
        and isinstance(coverage, dict)
        and coverage.get("status") == "measured"
        and isinstance(reasons, list)
        and "language route: unknown_language_bootstrap" in reasons
    ):
        engine = page_metadata.get("inference_backend")
        if engine == "paddle" and (
            page_metadata.get("ocr_method") in {None, "PaddleOCR"}
            and page_metadata.get("ocr_confidence_basis") == "paddle_recognition_score"
        ):
            candidates = [
                match for reason in reasons if isinstance(reason, str)
                for match in [re.fullmatch(
                    rf"unconfirmed text-language candidate {re.escape(expected)} \(([0-9]+\.[0-9]{{2}})\)",
                    reason,
                )]
                if match is not None
            ]
            if len(candidates) == 1 and 0.50 <= float(candidates[0].group(1)) < 0.70:
                return "reviewable_unknown_detection"
        if engine == "tesseract" and (
            page_metadata.get("ocr_method") == "Tesseract"
            and page_metadata.get("ocr_confidence_basis") == "tesseract_word_confidence_percent_scaled"
            and page_metadata.get("status") == "fallback"
            and isinstance(page_metadata.get("ocr_confidence"), (int, float))
            and not isinstance(page_metadata["ocr_confidence"], bool)
            and 0.50 <= page_metadata["ocr_confidence"] <= 1.0
            and isinstance(page_metadata.get("text_length"), int)
            and page_metadata["text_length"] > 0
        ):
            return "reviewable_unknown_detection"
    return "invalid"


def _validate_manifest(manifest: dict, mapping: dict) -> None:
    schema = manifest.get("schema_version")
    if schema not in {"ocr-rescue-corpus-v1", "ocr-rescue-corpus-v2"}:
        raise ValueError("unsupported corpus manifest schema")
    if not isinstance(mapping, dict):
        raise ValueError("outputs mapping must be an object")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("corpus manifest must contain cases")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("corpus case IDs must be unique nonempty strings")
    for case in cases:
        if schema == "ocr-rescue-corpus-v2" and type(case.get("require_exact_text")) is not bool:
            raise ValueError(f"case {case['id']} requires an explicit exact-text policy boolean")
        if schema == "ocr-rescue-corpus-v1" and "require_exact_text" in case:
            raise ValueError(f"case {case['id']} exact-text policy requires corpus v2")
        if not isinstance(case.get("pages"), list) or not case["pages"]:
            raise ValueError(f"case {case['id']} has no pages")
        if case.get("source_kind", "pdf") not in {"pdf", "png"}:
            raise ValueError(f"case {case['id']} has unsupported source kind")
        if not isinstance(case.get("source_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", case["source_sha256"]):
            raise ValueError(f"case {case['id']} has malformed source hash")
        for page in case["pages"]:
            if not isinstance(page, dict) or not isinstance(page.get("expected_text"), str):
                raise ValueError(f"case {case['id']} has malformed page truth")
    requirements = {
        "max_cer": (0, 1), "max_wer": (0, 1), "max_visual_mae": (0, 1),
        "min_foreground_recall": (0, 1), "max_size_error_pt": (0, 100),
        "min_region_overlap": (0, 1), "max_missing_tokens": (0, 10000),
        "max_unexpected_tokens": (0, 10000),
    }
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(requirements):
        raise ValueError("corpus thresholds missing or malformed")
    for name, (minimum, maximum) in requirements.items():
        value = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"malformed threshold: {name}")
        if name.endswith("_tokens") and not isinstance(value, int):
            raise ValueError(f"malformed threshold: {name}")


def _validate_case(case: dict, result: dict, manifest_dir: Path, mapping_dir: Path, thresholds: dict) -> dict:
    failures: list[str] = []
    pages: list[dict] = []
    source_kind = case.get("source_kind", "pdf")
    source_path = (manifest_dir / (case["source_pdf"] if source_kind == "pdf" else case["source_image"])).resolve()
    if not source_path.is_file() or _sha256(source_path) != case["source_sha256"]:
        raise ValueError(f"source hash mismatch or missing source: {case['id']}")
    for image in case.get("source_images", []):
        image_path = (manifest_dir / image["path"]).resolve()
        if not image_path.is_file() or _sha256(image_path) != image["sha256"]:
            raise ValueError(f"source image hash mismatch: {case['id']}")
    pdf_path = (mapping_dir / result["pdf"]).resolve()
    text_path = (mapping_dir / result["text"]).resolve()
    metadata = _metadata(result.get("metadata"), mapping_dir)
    observed_pdf_pages: list[str] = []
    source = fitz.open(source_path) if source_kind == "pdf" else None
    try:
      with fitz.open(pdf_path) as output:
        if (source is not None and source.page_count != len(case["pages"])) or output.page_count != len(case["pages"]):
            raise ValueError("source/output/manifest page count differs")
        text_pages = _page_texts(text_path, output.page_count, metadata)
        for index, truth in enumerate(case["pages"]):
            page = output[index]
            original = _render(source[index]) if source is not None else _render_image(source_path, truth["size_pt"])
            expected = normalize(truth["expected_text"])
            pdf_text = normalize(page.get_text("text"))
            observed_pdf_pages.append(pdf_text)
            txt_text = normalize(text_pages[index])
            cer = _ratio(expected, pdf_text)
            wer = _ratio(expected, pdf_text, words=True)
            txt_cer = _ratio(expected, txt_text)
            parity = pdf_text == txt_text
            size_error = max(
                abs(page.rect.width - truth["size_pt"][0]),
                abs(page.rect.height - truth["size_pt"][1]),
            )
            try:
                visual = visual_metrics(original, page)
            except ValueError:
                visual = {"rgb_mae": 1.0, "foreground_recall_1px": 0.0}
            expected_tokens = Counter(re.findall(r"\w+", expected, re.UNICODE))
            observed_tokens = Counter(re.findall(r"\w+", pdf_text, re.UNICODE))
            missing_tokens = list((expected_tokens - observed_tokens).elements())
            unexpected_tokens = list((observed_tokens - expected_tokens).elements())
            missing_regions = []
            misaligned_regions = []
            placements = []
            for region in truth.get("regions", []):
                marker = region["text"]
                expected_rect = fitz.Rect(region["rect_pt"])
                matches = page.search_for(marker)
                best = max((_overlap_fraction(expected_rect, match) for match in matches), default=0.0)
                placements.append({"text": marker, "best_expected_area_overlap": round(best, 6), "match_count": len(matches)})
                if not matches:
                    missing_regions.append(marker)
                elif best < thresholds["min_region_overlap"]:
                    misaligned_regions.append(marker)
            duplicate_tokens = []
            for region in truth.get("regions", []):
                token = normalize(region["text"])
                allowed = len(re.findall(rf"(?<!\w){re.escape(token)}(?!\w)", expected))
                observed = len(re.findall(rf"(?<!\w){re.escape(token)}(?!\w)", pdf_text))
                if observed > allowed:
                    duplicate_tokens.append(region["text"])
            page_failures = []
            if cer > thresholds["max_cer"] or wer > thresholds["max_wer"] or txt_cer > thresholds["max_cer"]:
                page_failures.append("text_accuracy")
            if case.get("require_exact_text") is True and (pdf_text != expected or txt_text != expected):
                page_failures.append("exact_text")
            if not parity:
                page_failures.append("pdf_txt_parity")
            if size_error > thresholds["max_size_error_pt"]:
                page_failures.append("physical_size")
            if missing_regions:
                page_failures.append("missing_regions")
            if misaligned_regions:
                page_failures.append("search_alignment")
            if duplicate_tokens:
                page_failures.append("duplicate_text")
            if len(missing_tokens) > thresholds["max_missing_tokens"]:
                page_failures.append("missing_tokens")
            if len(unexpected_tokens) > thresholds["max_unexpected_tokens"]:
                page_failures.append("unexpected_tokens")
            if visual["rgb_mae"] > thresholds["max_visual_mae"] or visual["foreground_recall_1px"] < thresholds["min_foreground_recall"]:
                page_failures.append("visual_fidelity")
            page_metadata = _page_metadata(metadata, index, len(case["pages"]))
            language = truth.get("expected_language")
            language_disposition = _language_provenance_disposition(language, page_metadata)
            if language_disposition == "invalid":
                page_failures.append("language_provenance")
            if not expected and not (
                page_metadata.get("status") == "image_only"
                and page_metadata.get("needs_review") is True
                and isinstance(page_metadata.get("reason"), str)
                and page_metadata["reason"].strip()
            ):
                page_failures.append("blank_disposition")
            low_confidence = {"valid_retained": [], "valid_dropped": [], "noise_retained": [], "noise_rejected": []}
            for candidate in page_metadata.get("low_confidence_candidates", []):
                if int(candidate.get("page", 1)) != index + 1:
                    continue
                token = normalize(str(candidate.get("text", "")))
                if not token:
                    continue
                in_truth = token in expected_tokens
                in_output = token in observed_tokens
                label = (
                    "valid_retained" if in_truth and in_output else
                    "valid_dropped" if in_truth else
                    "noise_retained" if in_output else "noise_rejected"
                )
                low_confidence[label].append({"text": token, "confidence": candidate.get("confidence")})
            if low_confidence["valid_dropped"]:
                page_failures.append("valid_low_confidence_dropped")
            if low_confidence["noise_retained"]:
                page_failures.append("low_confidence_noise_retained")
            pages.append({
                "page": index + 1,
                "cer": round(cer, 6),
                "wer": round(wer, 6),
                "txt_cer": round(txt_cer, 6),
                "pdf_txt_parity": parity,
                "require_exact_text": case.get("require_exact_text", False),
                "size_error_pt": round(size_error, 6),
                "missing_regions": missing_regions,
                "misaligned_regions": misaligned_regions,
                "duplicate_tokens": duplicate_tokens,
                "missing_tokens": missing_tokens,
                "unexpected_tokens": unexpected_tokens,
                "search_placements": placements,
                "low_confidence": low_confidence,
                "language_provenance_disposition": language_disposition,
                **visual,
                "failures": page_failures,
            })
            failures.extend(f"page_{index + 1}:{reason}" for reason in page_failures)
    finally:
        if source is not None:
            source.close()
    combined_pdf_text = " ".join(observed_pdf_pages)
    forbidden_found = [token for token in case.get("forbidden_text", []) if normalize(token) in combined_pdf_text]
    if forbidden_found:
        failures.append("forbidden_old_text")
    if metadata.get("status") in {"failed", "cancelled"}:
        failures.append("output_disposition")
    return {
        "id": case["id"],
        "passed": not failures,
        "failures": failures,
        "forbidden_old_text_found": forbidden_found,
        "source_sha256": case["source_sha256"],
        "output_pdf_sha256": _sha256(pdf_path),
        "output_text_sha256": _sha256(text_path),
        "review_metadata": metadata,
        "pages": pages,
    }


def validate(manifest_path: Path, mapping_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, mapping)
    thresholds = manifest["thresholds"]
    results = []
    for case in manifest["cases"]:
        try:
            output = mapping[case["id"]]
            results.append(_validate_case(case, output, manifest_path.parent, mapping_path.parent, thresholds))
        except Exception as exc:
            results.append({"id": case["id"], "passed": False, "failures": [f"harness_error:{type(exc).__name__}:{exc}"]})
    return {
        "schema_version": "ocr-rescue-validation-v1",
        "passed": all(case["passed"] for case in results),
        "manifest_sha256": _sha256(manifest_path),
        "mapping_sha256": _sha256(mapping_path),
        "thresholds": thresholds,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path, help="case-to-output PDF/TXT mapping JSON")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.manifest, args.outputs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OCR rescue validation: {'PASS' if report['passed'] else 'FAIL'}; report={args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
